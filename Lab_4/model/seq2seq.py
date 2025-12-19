import torch
import torch.nn as nn
from typing import Tuple
from data_utils.vocab import Vocab
from .attention import BahdanauAttention, LuongAttention

class Seq2seq(nn.Module):
    def __init__(
            self,
            d_model: int = 256,
            num_layers: int = 3,
            dropout: float = 0.1,
            vocab: Vocab = None,
            attention_type: str = None
    ):
        super().__init__()

        self.vocab = vocab
        self.num_layers = num_layers
        self.hidden_size = 2*d_model
        if attention_type == "Bahdanau":
            self.attention = BahdanauAttention(hidden_size=self.hidden_size)
        elif attention_type == "Luong":
            self.attention = LuongAttention(hidden_size=self.hidden_size)
        else:
            self.attention = None
        decoder_input_size = self.hidden_size * 2 if self.attention is not None else self.hidden_size

        self.src_embedding = nn.Embedding(
            num_embeddings=vocab.total_src_tokens,
            embedding_dim=d_model,
            padding_idx=vocab.pad_idx
        )

        self.tgt_embedding = nn.Embedding(
            num_embeddings=vocab.total_tgt_tokens,
            embedding_dim=self.hidden_size,
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
            input_size=decoder_input_size,
            hidden_size=self.hidden_size,
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
        _, batch_size, dim = states.shape  # num_layers * 2, batch, hidden_size

        states = states.view(self.num_layers, 2, batch_size, dim)
        forward = states[:, 0, :, :]
        backward = states[:, 1, :, :]

        return torch.cat([forward, backward], dim=-1)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_x = self.src_embedding(x)

        # Encode
        enc_outputs, (h_enc, c_enc) = self.encoder(embedded_x)
        h_dec = self.combine_bidirectional(h_enc)
        c_dec = self.combine_bidirectional(c_enc)

        # Decode
        outputs = []
        curr_input = y[:, 0].unsqueeze(1)
        max_len = y.size(1) - 1
        
        for i in range(max_len):
            embedded_y = self.tgt_embedding(curr_input)

            if self.attention:
                context, _ = self.attention(h_dec[-1], enc_outputs)
                lstm_input = torch.cat([embedded_y, context.unsqueeze(1)], dim=-1)
            else: 
                lstm_input = embedded_y

            out, (h_dec, c_dec) = self.decoder(lstm_input, (h_dec, c_dec))
            logit = self.output_head(out)
            outputs.append(logit)
            curr_input = y[:, i+1].unsqueeze(1)

        return torch.cat(outputs, dim=1)

    def predict(self, x: torch.Tensor, max_len: int = 100):
        embedded_x = self.src_embedding(x)

        enc_outputs, (h_enc, c_enc) = self.encoder(embedded_x)
        h_dec = self.combine_bidirectional(h_enc)
        c_dec = self.combine_bidirectional(c_enc)

        y_i = torch.full((x.size(0), 1), self.vocab.bos_idx, dtype=torch.long, device=x.device)
        outputs = []
        for _ in range(max_len):
            embedded_y = self.tgt_embedding(y_i)

            if self.attention:
                context, _ = self.attention(h_dec[-1], enc_outputs)
                lstm_input = torch.cat([embedded_y, context.unsqueeze(1)], dim=-1)
            else: 
                lstm_input = embedded_y

            output, (h_dec, c_dec) = self.decoder(lstm_input, (h_dec, c_dec))

            logits = self.output_head(output.squeeze(1))
            y_i = logits.argmax(dim=-1, keepdim=True)
            outputs.append(y_i)
        return torch.cat(outputs, dim=1)