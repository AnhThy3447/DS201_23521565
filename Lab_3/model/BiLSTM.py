import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, tag_size, embed_dim=128, 
                 hidden_dim=256, num_layers=5, dropout=0.3):
        super(BiLSTM_NER, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, tag_size)

    def forward(self, x, lengths):
        embeds = self.dropout(self.embedding(x))
        # Pack padded sequence
        packed_embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        packed_output, _ = self.bilstm(packed_embeds)
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        outputs = self.dropout(outputs)
        return self.fc(outputs)