import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionalBias(nn.Module):
    """相对位置偏置模块，用于HSTU中的位置编码"""
    
    def __init__(self, max_seq_len: int):
        super().__init__()
        self._max_seq_len = max_seq_len
        self._w = nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02)
        )

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: 序列长度
        Returns:
            位置偏置矩阵 [seq_len, seq_len]
        """
        n = seq_len
        if n > self._max_seq_len:
            n = self._max_seq_len
            
        t = F.pad(self._w[:2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class HSTUSequentialTransductionUnit(nn.Module):
    """
    HSTU的核心组件 - 序列转导单元
    
    基于论文: Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
    简化版本，适配您的baseline模型
    """
    
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        dropout_ratio: float = 0.1,
        linear_activation: str = "silu",
        enable_relative_bias: bool = True,
        max_seq_len: int = 200,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.linear_dim = linear_hidden_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout_ratio = dropout_ratio
        self.linear_activation = linear_activation
        
        # UVQK线性变换 - HSTU的核心创新
        # U: 门控向量, V: 值向量, Q: 查询向量, K: 键向量
        self.uvqk_projection = nn.Linear(
            embedding_dim,
            linear_hidden_dim * 2 * num_heads + attention_dim * num_heads * 2
        )
        
        # 输出投影层，支持concat_ua模式
        self.output_projection = nn.Linear(
            linear_hidden_dim * num_heads,
            embedding_dim
        )
        
        # Layer Normalization
        self.input_layernorm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.attn_layernorm = nn.LayerNorm(linear_hidden_dim * num_heads, eps=1e-6)
        
        # 相对位置偏置
        self.rel_pos_bias = RelativePositionalBias(max_seq_len) if enable_relative_bias else None
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.uvqk_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.uvqk_projection.bias)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len, seq_len]
            
        Returns:
            输出张量 [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 输入层归一化
        normed_x = self.input_layernorm(x)
        
        # 2. UVQK投影 - HSTU的核心
        uvqk_output = self.uvqk_projection(normed_x)
        
        # 3. 激活函数
        if self.linear_activation == "silu":
            uvqk_output = F.silu(uvqk_output)
        elif self.linear_activation == "relu":
            uvqk_output = F.relu(uvqk_output)
        
        # 4. 分割UVQK
        u, v, q, k = torch.split(
            uvqk_output,
            [
                self.linear_dim * self.num_heads,
                self.linear_dim * self.num_heads, 
                self.attention_dim * self.num_heads,
                self.attention_dim * self.num_heads,
            ],
            dim=-1
        )
        
        # 5. 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.linear_dim)
        
        # 6. HSTU注意力计算 - 使用SiLU而非Softmax
        attention_scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
        
        # 7. 添加相对位置偏置
        if self.rel_pos_bias is not None:
            pos_bias = self.rel_pos_bias(seq_len).to(x.device)
            attention_scores = attention_scores + pos_bias.unsqueeze(0).unsqueeze(0)
        
        # 8. HSTU特有：使用SiLU激活而非Softmax
        attention_scores = F.silu(attention_scores) / seq_len
        
        # 9. 应用注意力掩码
        if attention_mask is not None:
            # 扩展掩码维度以匹配多头
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # [B, 1, S, S]
            attention_scores = attention_scores * attention_mask
        
        # 10. 计算注意力输出
        attn_output = torch.einsum('bhqk,bkhd->bqhd', attention_scores, v)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, -1)
        
        # 11. 注意力输出层归一化
        normalized_attn = self.attn_layernorm(attn_output)
        
        # 12. 门控机制 - HSTU的另一个关键创新
        gated_output = u * normalized_attn
        
        # 13. 输出投影和残差连接
        output = F.dropout(gated_output, p=self.dropout_ratio, training=self.training)
        output = self.output_projection(output) + x
        
        return output


class HSTUBlock(nn.Module):
    """
    HSTU块，包含序列转导单元
    替换原有的Transformer块
    """
    
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        dropout_ratio: float = 0.1,
        linear_activation: str = "silu",
        enable_relative_bias: bool = True,
        max_seq_len: int = 200,
    ):
        super().__init__()
        
        self.hstu_unit = HSTUSequentialTransductionUnit(
            embedding_dim=embedding_dim,
            linear_hidden_dim=linear_hidden_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            dropout_ratio=dropout_ratio,
            linear_activation=linear_activation,
            enable_relative_bias=enable_relative_bias,
            max_seq_len=max_seq_len,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len, seq_len]
            
        Returns:
            输出张量 [batch_size, seq_len, embedding_dim]
        """
        return self.hstu_unit(x, attention_mask)


class HSTUEncoder(nn.Module):
    """
    HSTU编码器，包含多个HSTU块
    用于替换原baseline中的Transformer层
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        linear_hidden_dim: Optional[int] = None,
        attention_dim: Optional[int] = None,
        dropout_ratio: float = 0.1,
        linear_activation: str = "silu",
        enable_relative_bias: bool = True,
        max_seq_len: int = 200,
    ):
        super().__init__()
        
        # 默认参数设置
        if linear_hidden_dim is None:
            linear_hidden_dim = embedding_dim
        if attention_dim is None:
            attention_dim = embedding_dim // num_heads
            
        self.blocks = nn.ModuleList([
            HSTUBlock(
                embedding_dim=embedding_dim,
                linear_hidden_dim=linear_hidden_dim,
                attention_dim=attention_dim,
                num_heads=num_heads,
                dropout_ratio=dropout_ratio,
                linear_activation=linear_activation,
                enable_relative_bias=enable_relative_bias,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_blocks)
        ])
        
        self.final_layernorm = nn.LayerNorm(embedding_dim, eps=1e-6)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len, seq_len]
            
        Returns:
            编码后的张量 [batch_size, seq_len, embedding_dim]
        """
        # 通过所有HSTU块
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # 最终层归一化
        x = self.final_layernorm(x)
        
        return x