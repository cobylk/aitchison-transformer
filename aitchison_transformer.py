"""CLR/Aitchison Transformer implementation based on simplified clr_ops.

This uses the provided clr_ops style with proj_H, to_clr, from_clr operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.types import Tensor
from .base import ArchConfig, Architecture

# ------------------------- CLR ops ------------------------- #

def proj_H(z):
    """Project last dim onto H by subtracting mean."""
    return z - z.mean(dim=-1, keepdim=True)

def to_clr(p, eps=1e-6):
    """p simplex -> H"""
    z = torch.log(p + eps)
    return proj_H(z)

def from_clr(z):
    """H -> simplex"""
    return F.softmax(z, dim=-1)

def onehot_simplex(x, V, tau=1e-4):
    """Create smoothed one-hot simplex from token ids."""
    B, T = x.shape
    p = torch.full((B, T, V), tau / V, device=x.device, dtype=torch.float32)
    p.scatter_(2, x.unsqueeze(-1), 1.0 - tau + tau / V)
    return p

def causal_mask(B, T, device=None):
    """(B, 1, T, T) with 0 on and below diag, -inf above (additive mask)"""
    m = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    m = m.masked_fill(m == 1, float('-inf')).unsqueeze(0).unsqueeze(1)  # (1,1,T,T)
    return m.expand(B, 1, T, T)

# ------------------------- configuration ------------------------- #

class CLRTransformerConfig(ArchConfig):
    """Configuration for the CLR / Aitchison Transformer."""

    # Model dims - provide defaults
    vocab_size: int = 65
    d_model: int = 128
    max_seq_len: int = 1024
    
    n_heads: int = 4
    n_layers: int = 12
    d_ff: int = 512  # size of hidden in feed-forward

    # Geometry / numerical
    tau: float = 1e-4  # input one-hot smoothing
    epsilon: float = 1e-6  # for log safety & LN

    temperature: float = 1.0  # temperature for softmax in CLR space

    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-5  # not used but kept for compat

    seed: int = 42

# ------------------------- layer norm on H ------------------------- #

class HLayerNorm(nn.Module):
    """LN that preserves H"""
    def __init__(self, D, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(D))
        self.b = nn.Parameter(torch.zeros(D))
        self.eps = eps
    def forward(self, z):
        var = z.var(dim=-1, keepdim=True, unbiased=False)
        z = z / torch.sqrt(var + self.eps)
        z = z * self.g + self.b
        return proj_H(z)

# ------------------------- attention ------------------------- #

class ClrSelfAttention(nn.Module):
    def __init__(self, D, n_heads=4, d_head=None, dropout=0.1):
        super().__init__()
        d_head = d_head or (D // n_heads)
        self.nh, self.dk = n_heads, d_head
        self.q = nn.Linear(D, n_heads*d_head, bias=True)
        self.k = nn.Linear(D, n_heads*d_head, bias=True)
        self.v = nn.Linear(D, n_heads*d_head, bias=True)
        self.o = nn.Linear(n_heads*d_head, D, bias=True)
        self.drop = nn.Dropout(dropout)
        for m in [self.q, self.k, self.v, self.o]:
            nn.init.xavier_uniform_(m.weight, gain=0.8)
            nn.init.zeros_(m.bias)
            
    def forward(self, p, attn_mask=None):         # p: (B,T,D) simplex
        B, T, D = p.shape
        z = to_clr(p)                              # (B,T,D) in H
        q = proj_H(self.q(z)).view(B, T, self.nh, self.dk).transpose(1, 2)  # (B,H,T,dk)
        k = proj_H(self.k(z)).view(B, T, self.nh, self.dk).transpose(1, 2)
        v = proj_H(self.v(z)).view(B, T, self.nh, self.dk).transpose(1, 2)  # values in H
        att = (q @ k.transpose(-1, -2)) / (self.dk ** 0.5)               # (B,H,T,T)
        if attn_mask is not None:                    # attn_mask: (B,1,T,T) additive
            att = att + attn_mask
        w = F.softmax(att, dim=-1)
        w = self.drop(w)
        z_out = w @ v                                                  # (B,H,T,dk)
        z_out = z_out.transpose(1, 2).contiguous().view(B, T, self.nh*self.dk)
        z_out = proj_H(self.o(z_out))
        return from_clr(z_out)                                         # back to simplex

# ------------------------- feed-forward ------------------------- #

class ClrFFN(nn.Module):
    def __init__(self, D, mult=4, dropout=0.1, nonlin='gelu'):
        super().__init__()
        H1 = mult*D
        self.norm = HLayerNorm(D)
        self.l1 = nn.Linear(D, H1)
        self.l2 = nn.Linear(H1, D)
        self.act = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'tanh': nn.Tanh()}[nonlin]
        self.drop = nn.Dropout(dropout)
        for m in [self.l1, self.l2]:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
            
    def forward(self, p):
        z = to_clr(p)
        z = self.norm(z)
        z = proj_H(self.l1(z))
        z = self.act(z)
        z = self.drop(z)
        z = proj_H(self.l2(z))
        return from_clr(z)

# ------------------------- transformer block ------------------------- #

class ClrBlock(nn.Module):
    def __init__(self, D, n_heads=4, dropout=0.1, temperature=1.0):
        super().__init__()
        self.norm1 = HLayerNorm(D)
        self.attn = ClrSelfAttention(D, n_heads, dropout=dropout)
        self.norm2 = HLayerNorm(D)
        self.ffn = ClrFFN(D, mult=4, dropout=dropout)

        self.temperature = temperature
        
    def aitchison_resid(self, p, q):
        z = to_clr(p)
        z2 = to_clr(q)
        z_ = z + z2
        return from_clr(proj_H(z_))
    
    def apply_temperature(self, p):
        """Apply temperature in CLR space then convert back to simplex."""
        if self.temperature == 1.0:
            return p
        z = to_clr(p)
        z_temp = z / self.temperature
        return from_clr(proj_H(z_temp))
    
    def forward(self, p, attn_mask=None):
        p = self.aitchison_resid(p, self.attn(from_clr(self.norm1(to_clr(p))), attn_mask))
        p = self.aitchison_resid(p, self.ffn(from_clr(self.norm2(to_clr(p)))))
        p = self.apply_temperature(p)
        return p

# ------------------------- top-level model ------------------------- #

class CLRTransformer(Architecture):
    """CLR/Aitchison Transformer."""

    def __init__(self, config: CLRTransformerConfig):
        super().__init__(config)
        torch.manual_seed(config.seed)

        self.V = config.vocab_size
        self.D = config.d_model
        self.block = config.max_seq_len
        self.tau = config.tau

        # Δ_V -> Δ_D input layer (Aitchison-linear)
        self.in_lin = nn.Linear(self.V, self.D, bias=True)    # acts on log p via tying (see forward)
        self.pos_H = nn.Parameter(torch.zeros(1, self.block, self.D))  # mean-zero enforced in forward
        self.blocks = nn.ModuleList([
            ClrBlock(self.D, config.n_heads, config.dropout_rate, temperature=config.temperature) 
            for _ in range(config.n_layers)
        ])
        # Δ_D -> logits over V
        self.out_lin = nn.Linear(self.D, self.V, bias=False)
        
        nn.init.xavier_uniform_(self.in_lin.weight, gain=0.8)
        nn.init.zeros_(self.in_lin.bias)
        nn.init.xavier_uniform_(self.out_lin.weight, gain=0.8)

    @classmethod
    def get_config_class(cls) -> type[CLRTransformerConfig]:
        return CLRTransformerConfig
    
    def set_temperature(self, temperature: float) -> None:
        """Set temperature for all transformer blocks."""
        for block in self.blocks:
            block.temperature = temperature

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        x = input_ids
        B, T = x.shape
        V, D = self.V, self.D
        device = x.device
        
        attn_mask = causal_mask(B, T, device=device)  # proper causal masking
        
        p0 = onehot_simplex(x, V, tau=self.tau)     # (B,T,V) simplex
        z0 = to_clr(p0)                              # log then center
        z0 = proj_H(self.in_lin(z0))                # H_V -> H_D
        pos = proj_H(self.pos_H[:, :T, :])
        p = from_clr(z0 + pos)                      # back to simplex
        
        for blk in self.blocks:
            p = blk(p, attn_mask=attn_mask)
        # readout: H_D -> logits over vocab
        z = to_clr(p)
        logits = self.out_lin(z)                    # (B,T,V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss