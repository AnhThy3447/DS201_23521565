import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size=256):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size

        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False) 
        self.v = nn.Linear(hidden_size, 1, bias=False) 

    def forward(self, decoder_hidden, encoder_outputs):
        query = self.W_q(decoder_hidden).unsqueeze(1)
        keys = self.W_k(encoder_outputs)

        # Additive attention
        energy = torch.tanh(query + keys) 
        scores = self.v(energy).squeeze(2)

        attention_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class LuongAttention(nn.Module):
    def __init__(self, hidden_size=256):
        super(LuongAttention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        query = decoder_hidden.unsqueeze(1) 
        keys = self.W(encoder_outputs) 

        # Multiplicative attention
        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) 

        attention_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights