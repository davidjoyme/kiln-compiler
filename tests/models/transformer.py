# File: tests/models/transformer.py
import math, torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    """GPT-2 style pre-LN transformer block (inference)."""
    def __init__(self, d=64, nh=4, T=16):
        super().__init__()
        self.nh = nh
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.c_attn = nn.Linear(d, 3 * d); self.c_proj = nn.Linear(d, d)
        self.fc = nn.Linear(d, 4 * d); self.proj = nn.Linear(4 * d, d)
        mask = torch.full((T, T), float("-inf")).triu(1)
        self.register_buffer("mask", mask)
    def attn(self, x):
        T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=1)
        hd = C // self.nh
        q = q.view(T, self.nh, hd).transpose(0, 1)
        k = k.view(T, self.nh, hd).transpose(0, 1)
        v = v.view(T, self.nh, hd).transpose(0, 1)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = att + self.mask[:T, :T]
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(0, 1).contiguous().view(T, C)
        return self.c_proj(y)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.proj(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))
        return x