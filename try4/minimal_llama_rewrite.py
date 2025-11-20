from typing import Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import TransformersKwargs,auto_docstring, can_return_tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.cache_utils import Cache,DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from transformers.masking_utils import create_causal_mask
from typing import Callable
from transformers.processing_utils import Unpack
from transformers.generation import GenerationMixin
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaDecoderLayer,
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward
)
from safetensors.torch import load_file as safetensors_load 
from KV_Cache import RecordStorage

class MyLlamaConfig(LlamaConfig):
    """扩展配置，添加自定义参数"""
    def __init__(
        self,
        my_attention_param: float = 0.1,
        use_custom_attention: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.my_attention_param = my_attention_param
        self.use_custom_attention = use_custom_attention

class MyLlamaAttention(LlamaAttention):
    """只重写forward函数的Attention类"""
    def __init__(self, config: MyLlamaConfig, layer_idx: Optional[int] = None):
        # 完全继承父类的初始化，保持权重名称一致
        super().__init__(config, layer_idx)
        #从 config 对象中获取名为 "my_attention_param" 的属性。
        #如果 config 中没有这个属性，就使用默认值 0.1。
        self.my_attention_param = getattr(config, "my_attention_param", 0.1)
        self.flag = getattr(config, "flag", -1)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        # print("Your own Attention:", self.flag)
        
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        flags: Optional[dict] = None,
        KV_cache: Optional["RecordStorage"] = None,
        best_idx: Optional[int] = None,                     
        max_k: Optional[int] = None, 
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        #--------------用：match_KV + 拼接--------------#
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        #-------------当前prompt的K,V已经计算完成，存储下来后续使用，先不rope-----------#
        #------------存：KV_cache--------#
        if flags["is_prefill"]==0 and 0 <= self.layer_idx <= 15 :
            KV_cache.add(layer_id=self.layer_idx, 
                         K_states=key_states, 
                         V_states=value_states)
        #-----------------------------------------------------------------#
        

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        
        #----------标记prefill阶段结束-------------#
        if self.layer_idx==15:
            flags["is_prefill"]=1  
        #------------------------------------------------------------------#        

        #同一prompt的KV缓存
        if past_key_values is not None:   #对象只要创建了就可以
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class MyLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MyLlamaConfig, layer_idx: int):
        # 正确调用父类初始化，然后覆盖 self_attn
        super().__init__(config, layer_idx)
        # 用自定义 attention 覆盖（保持权重命名）
        self.self_attn = MyLlamaAttention(config=config, layer_idx=layer_idx)
        self.layer_idx = layer_idx
        
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        flags: Optional[dict] = None,
        KV_cache: Optional["RecordStorage"] = None,
        best_idx: Optional[int] = None,                     # 可选的索引/控制参数
        max_k: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # if(flags["is_prefill"]==0 and self.layer_idx==0):
        #     print("Your own LlamaDecoderLayer:",flags["is_prefill"])
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            flags=flags,
            KV_cache=KV_cache,
            best_idx=best_idx, 
            max_k=max_k,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class MyLlamaModel(LlamaModel):
    def __init__(self, config: MyLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MyLlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.post_init()
        
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        flags: Optional[dict] = None,
        KV_cache: Optional["RecordStorage"] = None,
        best_idx: Optional[int] = None,                     # 可选的索引/控制参数
        max_k: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """
        Forward pass for MyLlamaModel.
        
        Args:
            input_ids (`torch.LongTensor`, *optional*):
                Input token IDs.
            attention_mask (`torch.FloatTensor`, *optional*):
                Mask to avoid performing attention on padding token indices.
            ...
            flags (dict, optional): 可变字典，用于在 forward 层之间传递状态，例如 "is_prefill"。
            该 dict 是可变的，内部修改会反映到调用方。
            KV_cache (RecordStorage, optional): 自定义 KV cache，用于存/取 attention KV。
            best_idx (int, optional): 用户自定义索引。
            max_k (int, optional): 用户自定义参数。
            **kwargs:
                Additional arguments passed to submodules.
        Returns:
            [`BaseModelOutputWithPast`]: Model outputs with hidden states and optional past key values.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # if(flags["is_prefill"]==0):
        #     print("Your own LlamaModel:",flags["is_prefill"])
            
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                flags=flags,
                KV_cache=KV_cache,
                best_idx=best_idx, 
                max_k=max_k,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        #一次前向传播（16层transformer完成）
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MyLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: MyLlamaConfig):
        super().__init__(config)
        # 替换原始模型为自定义模型，保证使用 MyLlamaDecoderLayer 和 MyLlamaAttention
        self.model = MyLlamaModel(config)
        # print("MyLlamaForCausalLM config:", config)
    
    #类方法  
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        print("Loading MyLlamaForCausalLM from pretrained:", pretrained_model_name_or_path)
        #model_args: ()
        #kwargs: {'custom_config': {'my_attention_param': 0.1, 'use_custom_attention': True, 'flag': 1}, 
        #         'torch_dtype': torch.float16, 
        #         'device_map': 'auto'}
        custom_config_params = kwargs.pop("custom_config", {}) or {}
        map_location = kwargs.pop("map_location", "cpu")
        torch_dtype = kwargs.pop("torch_dtype", torch.float32)
        device_map = kwargs.pop("device_map", None)
        
        # load base config
        base_cfg = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
        cfg_dict = base_cfg.to_dict()
        #把 custom_config_params（期望为 dict）合并到cfg_dict，并且以 custom_config_params 的值
        #覆盖 cfg_dict 中相同键的原值。也就是说，用户提供的值优先级比磁盘上的 config.json 更高。
        cfg_dict.update(custom_config_params)
        custom_cfg = MyLlamaConfig(**cfg_dict)
        #custom_cfg：第一个位置参数，通常是 MyLlamaConfig 实例，也就是传给构造函数的 config。
        #调用本类的构造函数
        model = cls(custom_cfg, *model_args, **kwargs)
        if getattr(model.config.get_text_config(decoder=True), "tie_word_embeddings", True):
            # print("tie")
            pass
        else :
            print("no tie")
        # 然后把权重加载到指定 dtype
        if torch_dtype != torch.float32:
            model = model.to(torch_dtype)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        sd = None
        candidate_safetensors = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        candidate_pt = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if os.path.isfile(pretrained_model_name_or_path):
            path = pretrained_model_name_or_path
            if path.endswith(".safetensors"):
                sd = safetensors_load(path)
            else:
                sd = torch.load(path, map_location=map_location)
        else:
            if os.path.exists(candidate_safetensors):
                sd = safetensors_load(candidate_safetensors)
                print("Loaded safetensors from", candidate_safetensors)
            elif os.path.exists(candidate_pt):
                sd = torch.load(candidate_pt, map_location=map_location)
            else:
                from transformers import AutoModel
                hf_model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, **{"torch_dtype": kwargs.get("torch_dtype", None)})
                sd = hf_model.state_dict()
        
        model.tie_weights()
        if sd is not None:
            missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
            # print("load_state_dict: missing keys:", missing_keys)
            # print("load_state_dict: unexpected keys:", unexpected_keys)
        else:
            print("Warning: no state_dict loaded (sd is None)")
        print("Hello, MyLlamaForCausalLM 加载完成。")
        return model
        
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        flags: Optional[dict] = None,
        KV_cache: Optional["RecordStorage"] = None,
        best_idx: Optional[int] = None,                     # 可选的索引/控制参数
        max_k: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for MyLlamaModel.
        
        Args:
            input_ids (`torch.LongTensor`, *optional*):
                Input token IDs.
            attention_mask (`torch.FloatTensor`, *optional*):
                Mask to avoid performing attention on padding token indices.
            ...
            flags (dict, optional): 可变字典，用于在 forward 层之间传递状态，例如 "is_prefill"。
            该 dict 是可变的，内部修改会反映到调用方。
            KV_cache (RecordStorage, optional): 自定义 KV cache，用于存/取 attention KV。
            best_idx (int, optional): 用户自定义索引。
            max_k (int, optional): 用户自定义参数。
            **kwargs:
                Additional arguments passed to submodules.
        Returns:
            [`BaseModelOutputWithPast`]: Model outputs with hidden states and optional past key values.
        """
        #模型调用返回
        #last_hidden_state: Optional[torch.FloatTensor] 
        #past_key_values: Optional[Cache] 
        #hidden_states: Optional[tuple[torch.FloatTensor, ...]]
        #attentions: Optional[tuple[torch.FloatTensor, ...]]
        # if(flags["is_prefill"]==0):
        #     print("Your own LlamaForCausalLM:",flags["is_prefill"])
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            flags=flags,
            KV_cache=KV_cache,
            best_idx=best_idx, 
            max_k=max_k,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )