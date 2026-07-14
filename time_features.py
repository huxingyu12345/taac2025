import datetime
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class TimeFeatureProcessor:
    """
    时间特征处理器，用于提取时间差、hour、week等时间特征
    """
    
    def __init__(self, 
                 time_diff_buckets: int = 100,
                 hour_buckets: int = 24,
                 week_buckets: int = 7,
                 enable_relative_time: bool = True):
        """
        Args:
            time_diff_buckets: 时间差分桶数量
            hour_buckets: 小时分桶数量 (0-23)
            week_buckets: 星期分桶数量 (0-6)
            enable_relative_time: 是否启用相对时间特征
        """
        self.time_diff_buckets = time_diff_buckets
        self.hour_buckets = hour_buckets
        self.week_buckets = week_buckets
        self.enable_relative_time = enable_relative_time
        
    def extract_timestamp_features(self, timestamp: int) -> Dict[str, int]:
        """
        从时间戳提取时间特征
        
        Args:
            timestamp: Unix时间戳
            
        Returns:
            时间特征字典，包含hour和week
        """
        if timestamp == 0:
            return {
                'time_hour': 0,
                'time_week': 0,
            }
            
        try:
            dt = datetime.datetime.fromtimestamp(timestamp)
            hour = dt.hour  # 0-23
            week = dt.weekday()  # 0-6 (Monday is 0)
            
            return {
                'time_hour': hour + 1,  # 1-24, 0 reserved for padding
                'time_week': week + 1,  # 1-7, 0 reserved for padding
            }
        except (ValueError, OSError):
            # 无效时间戳的情况
            return {
                'time_hour': 0,
                'time_week': 0,
            }
    
    def compute_time_diff(self, current_timestamp: int, reference_timestamp: int) -> int:
        """
        计算时间差并分桶
        
        Args:
            current_timestamp: 当前时间戳
            reference_timestamp: 参考时间戳（通常是序列最后一个时间戳）
            
        Returns:
            时间差分桶ID
        """
        if current_timestamp == 0 or reference_timestamp == 0:
            return 0
            
        time_diff = abs(reference_timestamp - current_timestamp)
        
        # 时间差分桶：使用对数分桶
        if time_diff == 0:
            return 1
        
        # 以分钟为单位进行对数分桶
        time_diff_minutes = time_diff / 60.0
        bucket = min(int(math.log(time_diff_minutes + 1) * 10) + 1, self.time_diff_buckets)
        
        return bucket
    
    def extract_sequence_time_features(self, 
                                     sequence_timestamps: List[int],
                                     max_len: int) -> Dict[str, List[int]]:
        """
        为整个序列提取时间特征
        
        Args:
            sequence_timestamps: 序列中的时间戳列表
            max_len: 序列最大长度
            
        Returns:
            包含各种时间特征的字典
        """
        if not sequence_timestamps:
            return {
                'time_diff': [0] * max_len,
                'time_hour': [0] * max_len,
                'time_week': [0] * max_len,
            }
        
        # 获取参考时间戳（序列中的最后一个有效时间戳）
        reference_timestamp = sequence_timestamps[-1] if sequence_timestamps else 0
        
        time_diff_features = []
        time_hour_features = []
        time_week_features = []
        
        for timestamp in sequence_timestamps:
            # 时间差特征
            if self.enable_relative_time:
                time_diff = self.compute_time_diff(timestamp, reference_timestamp)
            else:
                time_diff = 0
            time_diff_features.append(time_diff)
            
            # 绝对时间特征
            time_feats = self.extract_timestamp_features(timestamp)
            time_hour_features.append(time_feats['time_hour'])
            time_week_features.append(time_feats['time_week'])
        
        # Padding到指定长度
        while len(time_diff_features) < max_len:
            time_diff_features.insert(0, 0)
            time_hour_features.insert(0, 0)
            time_week_features.insert(0, 0)
        
        # 截断到指定长度
        time_diff_features = time_diff_features[-max_len:]
        time_hour_features = time_hour_features[-max_len:]
        time_week_features = time_week_features[-max_len:]
        
        return {
            'time_diff': time_diff_features,
            'time_hour': time_hour_features,
            'time_week': time_week_features,
        }


class TimeFeatureEmbedding(nn.Module):
    """
    时间特征Embedding层
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 time_diff_buckets: int = 100,
                 hour_buckets: int = 25,  # 24 + 1 for padding
                 week_buckets: int = 8):  # 7 + 1 for padding
        super().__init__()
        
        self.time_diff_emb = nn.Embedding(time_diff_buckets + 1, embedding_dim, padding_idx=0)
        self.time_hour_emb = nn.Embedding(hour_buckets, embedding_dim, padding_idx=0)
        self.time_week_emb = nn.Embedding(week_buckets, embedding_dim, padding_idx=0)
        
        # 时间特征融合层
        self.time_feature_combiner = nn.Linear(embedding_dim * 3, embedding_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.time_diff_emb.weight)
        nn.init.xavier_uniform_(self.time_hour_emb.weight)
        nn.init.xavier_uniform_(self.time_week_emb.weight)
        nn.init.xavier_uniform_(self.time_feature_combiner.weight)
        nn.init.zeros_(self.time_feature_combiner.bias)
        
        # 确保padding embedding为0
        self.time_diff_emb.weight.data[0, :] = 0
        self.time_hour_emb.weight.data[0, :] = 0
        self.time_week_emb.weight.data[0, :] = 0
    
    def forward(self, 
                time_diff: torch.Tensor,
                time_hour: torch.Tensor,
                time_week: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_diff: 时间差特征 [batch_size, seq_len]
            time_hour: 小时特征 [batch_size, seq_len]
            time_week: 星期特征 [batch_size, seq_len]
            
        Returns:
            融合后的时间embedding [batch_size, seq_len, embedding_dim]
        """
        diff_emb = self.time_diff_emb(time_diff)  # [B, L, D]
        hour_emb = self.time_hour_emb(time_hour)  # [B, L, D]
        week_emb = self.time_week_emb(time_week)  # [B, L, D]
        
        # 拼接所有时间特征
        combined_time_emb = torch.cat([diff_emb, hour_emb, week_emb], dim=-1)  # [B, L, 3*D]
        
        # 通过线性层融合
        time_emb = self.time_feature_combiner(combined_time_emb)  # [B, L, D]
        
        return time_emb


def extract_time_features_from_sequence(sequence_data: List[Tuple], 
                                      processor: TimeFeatureProcessor,
                                      max_len: int) -> Dict[str, List[int]]:
    """
    从序列数据中提取时间特征
    
    Args:
        sequence_data: 序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp), ...]
        processor: 时间特征处理器
        max_len: 序列最大长度
        
    Returns:
        时间特征字典
    """
    timestamps = []
    for record in sequence_data:
        if len(record) >= 6:
            timestamp = record[5] if record[5] is not None else 0
            timestamps.append(timestamp)
        else:
            timestamps.append(0)
    
    return processor.extract_sequence_time_features(timestamps, max_len)