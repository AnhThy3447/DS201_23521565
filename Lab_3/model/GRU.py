import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class GRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=5,
                 dropout=0.5, output_dim=1):
        super(GRU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, lengths):
         embeds = self.dropout(self.embedding(x))

         # Pack padded sequence
         packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
   
         packed_outputs, hidden = self.gru(packed_embeds)
   
         hidden = self.dropout(hidden[-1])
         return self.fc(hidden)