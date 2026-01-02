import math
import torch
import torch.nn as nn
from .transformer_encoder import TransformerEncoder, PositionnalEncoding

class Transformer_CLS (nn.Module):
    def __init__(self, vocab_size, pad_idx, num_classes, d_model=256, d_ff=1024, n_heads=4,
                 n_layers=3, dropout=0.1):
        super(Transformer_CLS, self).__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.PE = PositionnalEncoding(d_model, dropout)
        self.encoder = TransformerEncoder(d_model, d_ff, n_heads, n_layers, dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.PE(x)

        # Encoder
        features = self.encoder(x, mask)   # (B, L, d_model)

        # Masked mean pooling
        mask = mask.unsqueeze(-1).float()   # (B, L, 1)
        pooled = (features * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        # Classification
        return self.classifier(self.dropout(pooled))      # (B, num_labels)


class Transformer_NER (nn.Module):
    def __init__(self, vocab_size, pad_idx, num_tags, d_model=256, d_ff=1024, n_heads=4,
                 n_layers=3, dropout=0.1):
        super(Transformer_NER, self).__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.PE = PositionnalEncoding(d_model, dropout)
        self.encoder = TransformerEncoder(d_model, d_ff, n_heads, n_layers, dropout)
        self.tag_classifier = nn.Linear(d_model, num_tags)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.PE(x)

        # Encoder
        features = self.encoder(x, mask)   # (B, L, d_model)
        return self.tag_classifier(self.dropout(features))      # (B, L, num_tags)