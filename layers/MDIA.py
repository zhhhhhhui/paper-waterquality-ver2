
import torch
import torch.nn as nn


class MDIA(nn.Module):
    def __init__(self, input_dim, heads=8, dropout=0.1):
        super(MDIA, self).__init__()
        self.heads = heads
        self.scale = input_dim ** -0.5
        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=False)
        self.fc = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.heads, input_dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_len, input_dim)
        output = self.dropout(self.fc(attn_output))
        return output
