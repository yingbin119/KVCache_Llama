from typing import List, Tuple, Optional
import torch
import logging

def kmp_prefix_function(pat: List[int]) -> List[int]:
    """构建 KMP 的部分匹配表（前缀函数）"""
    pi = [0] * len(pat)
    j = 0
    for i in range(1, len(pat)):
        while j > 0 and pat[i] != pat[j]:
            j = pi[j - 1]
        if pat[i] == pat[j]:
            j += 1
        pi[i] = j
    return pi


def longest_prefix_of_pattern_in_text(pattern: List[int], text: List[int]) -> Tuple[int, Optional[int]]:
    """
    找出 pattern（必须从头开始）在 text 中出现的最长连续匹配长度 max_k 及其起始位置 start。
    """
    if not pattern or not text:
        return 0, None

    pi = kmp_prefix_function(pattern)
    j = 0
    max_k, max_start = 0, None
    for i, x in enumerate(text):
        while j > 0 and x != pattern[j]:
            j = pi[j - 1]
        if x == pattern[j]:
            j += 1
            if j > max_k:
                max_k = j
                max_start = i - j + 1
            if j == len(pattern):
                j = pi[j - 1]
    return max_k, max_start


def find_prefix_matches(A, sequences: List[List[int]]) -> Tuple[int, int, Optional[int]]:
    """
    对每个 sequence（已有序列），找出它从第一个元素开始的最大前缀，
    在 A 中出现的最大连续匹配长度 max_k、起始位置 start，以及序列编号。
    
    返回：
        (best_idx, max_k, best_start)
    """
    if isinstance(A, torch.Tensor):        # <-- 自动处理 GPU 张量
        A = A.squeeze().cpu().tolist()
    
    logging.info("match 调用，待匹配的列表A: %s",A)
    logging.info("match 调用，匹配库sequences:")
    for seq in sequences:
        logging.info("%s", seq)
    # --- 忽略第一个元素进行匹配 ---
    #----第一个元素是128000，开始符号
    A_sub = A[1:]
    max_k, best_start, best_idx = 0, None, -1
    for idx, seq in enumerate(sequences):
        seq_sub = seq[1:]
        k, start = longest_prefix_of_pattern_in_text(seq_sub, A_sub)
        if k > max_k:
            max_k, best_start, best_idx = k, start, idx
    
     # --- 匹配结果修正 ---
    if best_start is not None:
        best_start += 1                  
    max_k += 1        

    return best_idx, max_k, best_start
