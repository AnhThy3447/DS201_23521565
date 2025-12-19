import torch
import torch.nn as nn
from typing import Tuple
from data_utils.vocab import Vocab

class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        vocab: Vocab = None
    ):
        super().__init__()

        self.vocab = vocab
        self.num_layers = num_layers

        self.src_embedding = nn.Embedding(
            num_embeddings=vocab.total_src_tokens,
            embedding_dim=d_model,
            padding_idx=vocab.pad_idx
        )

        self.tgt_embedding = nn.Embedding(
            num_embeddings=vocab.total_tgt_tokens,
            embedding_dim=2*d_model,
            padding_idx=vocab.pad_idx
        )

        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.decoder = nn.LSTM(
            input_size=2*d_model,
            hidden_size=2*d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        self.output_head = nn.Linear(
            in_features=2*d_model,
            out_features=vocab.total_tgt_tokens
        )
    
    def combine_bidirectional(self, states: torch.Tensor) -> torch.Tensor:
        num_layers_total, batch_size, dim = states.shape  # num_layers * 2, batch, hidden_size

        states = states.view(self.num_layers, 2, batch_size, dim)
        forward = states[:, 0, :, :]
        backward = states[:, 1, :, :]

        return torch.cat([forward, backward], dim=-1)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_x = self.src_embedding(x)
        
        # Encode
        _, (h_enc, c_enc) = self.encoder(embedded_x) 
        h_dec = self.combine_bidirectional(h_enc)
        c_dec = self.combine_bidirectional(c_enc)

        # Decode
        decoder_input = y[:, :-1]
        embedded_y = self.tgt_embedding(decoder_input)
        output, _ = self.decoder(embedded_y, (h_dec, c_dec))

        return self.output_head(output)
    
    def predict(self, x: torch.Tensor, max_len: int = 100):
        embedded_x = self.src_embedding(x)
        _, (h_enc, c_enc) = self.encoder(embedded_x)

        h_dec = self.combine_bidirectional(h_enc)
        c_dec = self.combine_bidirectional(c_enc)

        y_i = torch.zeros(x.size(0), 1).fill_(self.vocab.bos_idx)
        outputs = []
        for _ in range(max_len):
            embedded_y = self.tgt_embedding(y_i)
            output, (h_dec, c_dec) = self.decoder(embedded_y, (h_dec, c_dec))

            logits = self.output_head(output.squeeze(1))
            y_i = logits.argmax(dim=-1, keepdim=True)
            outputs.append(y_i)
        return torch.cat(outputs, dim=1)