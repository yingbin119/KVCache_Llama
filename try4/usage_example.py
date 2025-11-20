# usage_example.py
from minimal_llama_rewrite import MyLlamaForCausalLM, MyLlamaConfig
from transformers import AutoTokenizer
import torch
from pathlib import Path
import re

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'KeyWord')))
from find_keywords import OrderedKeywordExtractor
from match import find_prefix_matches
from KV_Cache import RecordStorage
from log import setup_file_logger   
import logging

setup_file_logger("kv_cache.log")  

MODEL_DIR = "/mnt/data4/yingbin/llama3_1B_16/models/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
WEIGHT_PATH = str(Path(MODEL_DIR) / "model.safetensors")

# 在 generated 中查找 prompt_ids 的最后位置（从左到右匹配）
def find_subsequence(seq, subseq):
    # 返回 subseq 在 seq 中的第一个起始索引，若无返回 -1
    if len(subseq) == 0:
        return -1
    seq = seq.tolist() if isinstance(seq, torch.Tensor) else list(seq)
    sub = subseq.tolist() if isinstance(subseq, torch.Tensor) else list(subseq)
    n, m = len(seq), len(sub)
    for s in range(n - m + 1):
        if seq[s:s+m] == sub:
            return s
    return -1

def test_pretrained_loading():
    """测试从本地加载预训练权重"""
    # 方法1：使用from_pretrained
    model = MyLlamaForCausalLM.from_pretrained(
        MODEL_DIR,  # 本地模型路径
        custom_config={
            "my_attention_param": 0.1,
            "use_custom_attention": True,
            "flag": 0
        },
        torch_dtype=torch.float16,  # 可选：使用半精度节省内存
        device_map="auto"  # 可选：自动设备映射
    )
    device = next(model.parameters()).device    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, 
                                              use_fast=False,
                                              local_files_only=True)
    
    # 测试推理 - 批量 prompts 版本
    prompts = [
        "对早期肺结节患者，哪些影像学特征和实验室结果有助于癌症筛查？",
        "乳腺肿块患者，影像学特征与血清指标联合分析有什么诊断价值？",
        "超声影像与乳腺肿块生物标志物如何判断良恶性？",
        "如何通过影像学和实验室指标综合鉴别早期肺癌与良性肺结节？",
        "如何通过影像和实验室检查评估乳腺结节的恶性风险？",
        "影像学与血液指标如何联合判断早期肺癌和良性结节？"
    ]  
    #提取关键词
    extractor = OrderedKeywordExtractor(local_dir='/home/xiaohai/yingbin1/MY_Llama3/KeyWord/models')
    for i, sentence in enumerate(prompts):
        keywords = extractor.get_ordered_keywords_only(sentence, top_n=9, method="combined")
        combined = "".join(keywords)
        prompts[i] = combined
    
    #前缀复用缓存器
    KV_cache = RecordStorage(max_size=4)
    # 使用 padding=True 让不同长度的 prompt 对齐为 batch
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  
    
    logging.info("提取关键词后初始的prompts: %s", prompts)
    for i, prompt in enumerate(prompts):
        logging.info("--- prompt {%s} ---",i)
        logging.info("input (keywords): %s", prompt)
        # 对 prompt 进行 tokenize（返回张量）
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)        
        # 将输入移动到模型所在设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print("tokenizer后的input编码:",inputs["input_ids"])
        # 兼容某些 tokenizer 不返回 attention_mask 的情况
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], device=device)
        # 计算输入真实长度（attention_mask 的和）
        input_length = int(inputs["attention_mask"].sum(dim=1).long().item())
        logging.info("input length: %s", input_length)
        
        #------------当前prompt的inputs_id和存储器中的比较---------------# 
        max_k, best_start, best_idx = 0, None, -1
        A=inputs["input_ids"]
        if len(KV_cache.storage)>0:
            _,inputs_id_list = KV_cache.check()
            best_idx, max_k, best_start = find_prefix_matches(A=A, sequences=inputs_id_list)
            print(f"当前prompt:{i}  最优匹配--best_idx:{best_idx}  max_k:{max_k}  best_start:{best_start}")
            # 根据前缀匹配调整当前prompt中token的位置
            if best_start is not None and best_idx != -1:
                inputs["input_ids"] = torch.cat([inputs["input_ids"][:,:1], 
                                                inputs["input_ids"][:,best_start:], 
                                                inputs["input_ids"][:,1:best_start]
                                                ],dim=1)
        #--------------------------------#
        print(f"调配后的关键词编码:",inputs["input_ids"])
        # 将关键词加入缓存
        KV_cache.add(keywords=prompt,
                     inputs_id=inputs["input_ids"])
        # 单条生成
        generated = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            flags={"is_prefill": 0},
            KV_cache=KV_cache,  # 传入 KV_cache
            best_idx=best_idx, 
            max_k=max_k
        )
        # generated 形状通常是 (1, seq_len_generated)，取第一行
        gen_ids = generated[0].cpu()
        total_len = gen_ids.size(0)
        # 找到 prompt 在生成序列中的起始位置（使用 find_subsequence）
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_ids = torch.tensor(prompt_ids, dtype=gen_ids.dtype)
        pos = find_subsequence(gen_ids, prompt_ids)
        if pos != -1:
            continuation_ids = gen_ids[pos + prompt_ids.size(0):]
        else:
            num_new = max(0, total_len - input_length)
            if num_new == 0:
                continuation_ids = torch.tensor([], dtype=gen_ids.dtype)
            else:
                continuation_ids = gen_ids[-num_new:]
        # 解码并做清理
        if continuation_ids.numel() == 0:
            decoded_cont = ""
        else:
            decoded_cont = tokenizer.decode(continuation_ids, skip_special_tokens=True)
        decoded_cont = re.sub(r'_xref[^;]*;', '', decoded_cont)
        logging.info("Generated: %s", decoded_cont)
    KV_cache.show()
    return model, tokenizer

def test_direct_creation():
    """测试直接创建模型（不加载预训练权重）"""
    config = MyLlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=4,  # 测试用少量层
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        my_attention_param=0.1,
        use_custom_attention=True
    )
    model = MyLlamaForCausalLM(config)
    print(f"模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

if __name__ == "__main__":
    # 测试直接创建
    # test_model = test_direct_creation()
    
    # 测试加载预训练权重（取消注释来测试）
    pretrained_model, tokenizer = test_pretrained_loading()