from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    d_vocab: int = 10_000
    d_model: int = 128
    d_mlp: int = 512
    n_heads: int = 4
    d_head: int = 32
    n_layers: int = 6
    act_fn: type[nn.Module] = nn.ReLU

class AttentionHead(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.W_q = nn.Parameter(torch.randn(cfg.d_model, cfg.d_head) * (1.0 / cfg.d_model ** 0.5))
        self.W_k = nn.Parameter(torch.randn(cfg.d_model, cfg.d_head) * (1.0 / cfg.d_model ** 0.5))
        self.W_v = nn.Parameter(torch.randn(cfg.d_model, cfg.d_head) * (1.0 / cfg.d_model ** 0.5))
        self.W_o = nn.Parameter(torch.randn(cfg.d_head, cfg.d_model) * (1.0 / cfg.d_head ** 0.5))
    
    def masking_matrix(self, n_context: int) -> torch.Tensor:
        m = torch.full((n_context, n_context), -torch.inf)
        return torch.triu(m, diagonal=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_context, d_model = x.shape  # Expects [n_context, d_model], e.g., [10, 128]
        Q = x @ self.W_q  # [n_context, d_head], e.g., [10, 32]
        K = x @ self.W_k  # [n_context, d_head]
        V = x @ self.W_v  # [n_context, d_head]
        scores = Q @ K.transpose(-2, -1) / (self.W_q.shape[-1] ** 0.5)  # [n_context, n_context], e.g., [10, 10]
        scores += self.masking_matrix(n_context)
        attn = F.softmax(scores, dim=-1)  # [n_context, n_context]
        out = attn @ V @ self.W_o  # [n_context, d_model], e.g., [10, 128]
        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.head_list = nn.ModuleList([AttentionHead(cfg) for _ in range(cfg.n_heads)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x  # [n_context, d_model]
        for h in self.head_list:
            output = output + h.forward(x)  # Residual connection
        return output

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.W_m_down = nn.Parameter(torch.randn(cfg.d_model, cfg.d_mlp) * (1.0 / cfg.d_model ** 0.5))
        self.W_m_up = nn.Parameter(torch.randn(cfg.d_mlp, cfg.d_model) * (1.0 / cfg.d_mlp ** 0.5))
        self.B = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.act_function = cfg.act_fn()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mlp = x @ self.W_m_down  # [n_context, d_mlp]
        x_mlp = self.act_function(x_mlp + self.B)
        x_mlp = x_mlp @ self.W_m_up  # [n_context, d_model]
        return x + x_mlp

class Transformer(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.d_vocab, cfg.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.MHA_layers = nn.ModuleList([MultiHeadedAttention(cfg) for _ in range(cfg.n_layers)])
        self.MLP_layers = nn.ModuleList([MLP(cfg) for _ in range(cfg.n_layers)])
        self.out_layer = nn.Linear(cfg.d_model, cfg.d_vocab)
        nn.init.normal_(self.out_layer.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = self.embedding(x)  # [n_context] -> [n_context, d_model]
        for MHA, MLP in zip(self.MHA_layers, self.MLP_layers):
            X = MHA.forward(X)
            X = MLP.forward(X)
        return self.out_layer(X)  # [n_context, d_vocab]