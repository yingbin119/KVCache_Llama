import os
import jieba.posseg as pseg
import jieba
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import re
from typing import List, Tuple

class OrderedKeywordExtractor:
    def __init__(self, model_name='BAAI/bge-small-zh-v1.5', local_dir=''):
        """
        Args:
            model_name: 用于语义相似度计算的模型
            local_dir: 本地模型存储目录
        """
        self.model_name = model_name
        self.local_dir = local_dir
        self.local_model_path = os.path.join(local_dir, model_name.replace('/', '_'))
        # 确保目录存在
        os.makedirs(local_dir, exist_ok=True)
        # 加载模型（智能方式：本地存在就加载，否则下载）
        self.model = self._load_model_smart()
        self.kw_model = KeyBERT(self.model)
        
    def _load_model_smart(self):
        """智能加载模型：本地存在就加载，否则下载并保存"""
        # 检查本地是否已有完整模型
        if self._is_model_cached_locally():
            print(f"从本地加载模型: {self.local_model_path}")
            return SentenceTransformer(self.local_model_path)
        else:
            print(f"下载模型中: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            print(f"保存模型到本地: {self.local_model_path}")
            model.save(self.local_model_path)
            return model
        
    def _is_model_cached_locally(self):
        """检查模型是否已缓存到本地目录"""
        if not os.path.exists(self.local_model_path):
            return False       
        # 检查必要的模型文件
        required_files = [
            'config.json',
            'sentence_bert_config.json',
            '1_Pooling/config.json'
        ]
        # 检查模型权重文件（.safetensors 或 .bin）
        weight_files = ['model.safetensors', 'pytorch_model.bin']
        
        for file in required_files:
            if not os.path.exists(os.path.join(self.local_model_path, file)):
                return False
        
        # 检查是否有权重文件
        has_weights = any(
            os.path.exists(os.path.join(self.local_model_path, f))
            for f in weight_files
        )
        
        return has_weights
    
    def extract_with_order(self, text: str, top_n: int , method: str = "combined") -> List[Tuple[str, float, int]]:
        """
        提取关键词并保持原顺序
        Args:
            text: 输入文本
            top_n: 返回关键词数量
            method: 提取方法 ("combined", "semantic", "position")
        
        Returns:
            List of (keyword, score, position)
        """
        if method == "combined":
            return self._combined_extract(text, top_n)
        elif method == "semantic":
            return self._semantic_extract(text, top_n)
        elif method == "position":
            return self._position_extract(text, top_n)
        else:
            raise ValueError("方法必须是 'combined', 'semantic' 或 'position'")
    
    def _position_extract(self, text: str, top_n: int) -> List[Tuple[str, float, int]]:
        """基于词性分析和位置的关键词提取"""
        words = pseg.cut(text)
        
        # 重要词性：名词、动词、形容词、专有名词等
        important_pos = {
            'n', 'v', 'a', 'nr', 'ns', 'nt', 'nz', 'vn', 'an',  # 名词、动词、形容词
            'x', 'eng'  # 非中文词、英文
        }
        
        keywords_with_pos = []
        
        for i, (word, flag) in enumerate(words):
            # 过滤条件：重要词性且长度合适
            if (flag in important_pos and 
                len(word) >= 1 and 
                self._is_valid_keyword(word)):
                
                # 计算简单分数（基于词性权重）
                score = self._calculate_pos_score(flag, word)
                start_pos = text.find(word)
                keywords_with_pos.append((word, score, start_pos))
        
        # 按位置排序
        keywords_with_pos.sort(key=lambda x: x[2])
        return keywords_with_pos[:top_n]
    
    def _semantic_extract(self, text: str, top_n: int) -> List[Tuple[str, float, int]]:
        """基于语义相似度的关键词提取"""
        # 使用KeyBERT提取关键词和分数
        keywords_with_scores = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='chinese',
            top_n=top_n * 2,  # 多提取一些用于后续过滤
            use_mmr=True,
            diversity=0.5
        )
        
        # 查找每个关键词在原句中的位置
        keywords_with_position = []
        for keyword, score in keywords_with_scores:
            # 找到关键词的起始位置
            start_pos = text.find(keyword)
            if start_pos != -1:  # 确保关键词在原文中
                keywords_with_position.append((keyword, score, start_pos))
        
        # 按位置排序
        keywords_with_position.sort(key=lambda x: x[2])
        
        return keywords_with_position[:top_n]
    
    def _combined_extract(self, text: str, top_n: int) -> List[Tuple[str, float, int]]:
        """结合语义和位置信息的综合提取方法"""
        # 获取语义关键词
        semantic_keywords = self._semantic_extract(text, top_n * 2)
        
        # 获取位置关键词
        position_keywords = self._position_extract(text, top_n * 2)
        
        # 合并并去重，优先保留语义分数高的
        keyword_dict = {}
        
        # 先添加语义关键词
        for kw, score, pos in semantic_keywords:
            if kw not in keyword_dict or score > keyword_dict[kw][0]:
                keyword_dict[kw] = (score, pos)
        
        # 添加位置关键词（如果不在字典中或分数更高）
        for kw, score, pos in position_keywords:
            if kw not in keyword_dict:
                keyword_dict[kw] = (score, pos)
        
        # 转换为列表并按位置排序
        combined_keywords = [
            (kw, score, pos) for kw, (score, pos) in keyword_dict.items()
        ]
        combined_keywords.sort(key=lambda x: x[2])
        
        return combined_keywords[:top_n]
    
    def _is_valid_keyword(self, word: str) -> bool:   #_position_extract
        """检查是否为有效关键词"""
        # 过滤停用词和无效字符
        invalid_patterns = [
            r'^\d+$',  # 纯数字
            r'^[^\u4e00-\u9fa5a-zA-Z]+$',  # 无中英文字符
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, word):
                return False
        
        # 简单停用词过滤
        stop_words = {'的', '了', '在', '是', '有', '和', '与', '及', '或'}
        if word in stop_words:
            return False
            
        return True
    
    def _calculate_pos_score(self, pos: str, word: str) -> float:  #_position_extract
        """基于词性计算初始分数"""
        pos_weights = {
            'n': 1.0, 'nr': 1.0, 'ns': 1.0, 'nt': 1.0, 'nz': 1.0,  # 名词
            'vn': 0.9, 'v': 0.8,  # 动词
            'an': 0.7, 'a': 0.7,  # 形容词
            'x': 0.6, 'eng': 0.6  # 其他
        }
        
        base_score = pos_weights.get(pos, 0.5)
        
        # 根据词长微调分数
        length_bonus = min(len(word) * 0.1, 0.3)
        
        return base_score + length_bonus
    
    def get_ordered_keywords_only(self, text: str, top_n: int = 10, method:str = "combined") -> List[str]:
        """只返回有序的关键词列表"""
        keywords = self.extract_with_order(text, top_n, method)
        return [kw for kw, score, pos in keywords]
    
    
# 使用示例
# def main():
#    # 现在模型会下载到当前目录下的 models 文件夹
#     extractor = OrderedKeywordExtractor(local_dir='./models')
#     print("模型存储位置:", os.path.abspath('./models'))
#     # 测试句子
#     test_sentences = [
#         "如何通过影像学和实验室指标综合鉴别早期肺癌与良性结节？",
#         "对影像学和实验室指标综合鉴别早期肺癌与良性结节的方法做出介绍。",
#         "介绍通过影像学和实验室指标综合鉴别早期肺癌与良性结节的方法。"
#     ]
#     method = "combined"
#     for sentence in test_sentences:
#         keywords = extractor.get_ordered_keywords_only(sentence, top_n=9, method=method)
#         print(f"原句: {sentence}")
#         print(f"关键词: {keywords}")
#         print(f"组合: {' '.join(keywords)}")

# if __name__ == "__main__":
#     main()