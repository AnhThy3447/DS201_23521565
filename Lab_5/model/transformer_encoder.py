import math
import copy
import torch
from torch import nn
from typing import Union
import torch.nn.functional as F
from typing import Optional, Tuple, Any


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor,
                mask: Optional[torch.ByteTensor] = None, dropout: Optional[nn.Dropout] = None) \
         -> Tuple[torch.Tensor, Any]:
        """
        Args:
            `query`: shape (batch_size, n_heads, max_len, d_q)
            `key`: shape (batch_size, n_heads, max_len, d_k)
            `value`: shape (batch_size, n_heads, max_len, d_v)
            `mask`: shape (batch_size, 1, 1, max_len)

        Returns:
            `weighted value`: shape (batch_size, n_heads, max_len, d_v)
            `weight matrix`: shape (batch_size, n_heads, max_len, max_len)
        """
        d_k = query.size(-1)  # d_k = d_model / n_heads
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # B*H*L*L
        if mask is not None:
            scores = scores.masked_fill(mask.eq(0), -1e9)
        p_attn = F.softmax(scores, dim=-1)  # B*H*L*L
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        # d_v = d_k
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.sdpa = ScaledDotProductAttention()
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor,
                mask: Optional[torch.ByteTensor] = None) -> torch.FloatTensor:
        """
        Args: 
            `query`: shape (batch_size, max_len, d_model)
            `key`: shape (batch_size, max_len, d_model)
            `value`: shape (batch_size, max_len, d_model)
            `mask`: shape (batch_size, max_len)
        
        Returns:
            shape (batch_size, max_len, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        batch_size = query.size(0)

        # Linear projections
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]

        # Apply attention 
        x, self.attn = self.sdpa(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.Linear1 = nn.Linear(d_model, d_ff)
        self.Linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.Linear2(self.dropout(F.relu(self.Linear1(x))))

            
class PositionnalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len=512):
        super(PositionnalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1, max_len, d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor, sublayer: Union[MultiHeadAttention, FeedForward]) -> torch.FloatTensor:
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

    
class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int = 1, n_layers: int = 1,
                 dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.multi_headed_attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.encoder_layer = EncoderLayer(d_model, self.multi_headed_attention, self.feed_forward, dropout)
        self.encoder = Encoder(self.encoder_layer, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        return self.encoder(x, mask)
    

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])