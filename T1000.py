from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@dataclass
class GPTConfig:
    d_vocab: int = 10_000
    d_model: int = 128
    d_mlp: int = 512
    n_heads: int = 4
    d_head: int = 32
    n_layers: int = 6
    act_fn: type[nn.Module] = nn.ReLU

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        norm_x = x / rms
        return norm_x * self.scale
    
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create constant positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-9.21 / d_model)) # 9.21 = log(10000)
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [seq_len, d_model] or [batch_size, seq_len, d_model]
        seq_len = x.size(-2)
        return self.pe[:seq_len, :]

class AttentionHead(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.W_q = nn.Parameter(torch.randn(cfg.d_model, cfg.d_head) * (1.0 / cfg.d_model ** 0.5))
        self.W_k = nn.Parameter(torch.randn(cfg.d_model, cfg.d_head) * (1.0 / cfg.d_model ** 0.5))
        self.W_v = nn.Parameter(torch.randn(cfg.d_model, cfg.d_head) * (1.0 / cfg.d_model ** 0.5))
        self.W_o = nn.Parameter(torch.randn(cfg.d_head, cfg.d_model) * (1.0 / cfg.d_head ** 0.5))
    
    def masking_matrix(self, n_context: int) -> torch.Tensor:
        m = torch.full((n_context, n_context), -torch.inf)
        if torch.cuda.is_available():
            return torch.triu(m, diagonal=1).to("cuda")
        else:
            return torch.triu(m, diagonal=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(0)
        n_context, d_model = x.shape
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        scores = Q @ K.transpose(-2, -1) / (self.W_q.shape[-1] ** 0.5)
        scores += self.masking_matrix(n_context)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V @ self.W_o
        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.head_list = nn.ModuleList([AttentionHead(cfg) for _ in range(cfg.n_heads)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = sum(h.forward(x) for h in self.head_list)
        return attn_output

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.W_m_down = nn.Parameter(torch.randn(cfg.d_model, cfg.d_mlp) * (1.0 / cfg.d_model ** 0.5))
        self.W_m_up = nn.Parameter(torch.randn(cfg.d_mlp, cfg.d_model) * (1.0 / cfg.d_mlp ** 0.5))
        self.B = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.act_function = cfg.act_fn()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mlp = x @ self.W_m_down
        x_mlp = self.act_function(x_mlp + self.B)
        x_mlp = x_mlp @ self.W_m_up
        return x_mlp

class Transformer(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embedding = SinusoidalPositionalEmbedding(cfg.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.MHA_layers = nn.ModuleList([MultiHeadedAttention(cfg) for _ in range(cfg.n_layers)])
        self.MLP_layers = nn.ModuleList([MLP(cfg) for _ in range(cfg.n_layers)])
        self.attn_norms = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.n_layers)])
        self.mlp_norms = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.out_layer = nn.Linear(cfg.d_model, cfg.d_vocab)
        nn.init.normal_(self.out_layer.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = self.embedding(x)
        X = X + self.pos_embedding(X)
        for i in range(len(self.MHA_layers)):
            attn_input = self.attn_norms[i](X)
            attn_output = self.MHA_layers[i](attn_input)
            X = X + attn_output
            mlp_input = self.mlp_norms[i](X)
            mlp_output = self.MLP_layers[i](mlp_input)
            X = X + mlp_output
        X = self.final_norm(X)
        return self.out_layer(X)
    
    # Need to save training tokenizer for later usage in generate function of model
    def save_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, text: str, max_new_tokens: int):
        self.eval()

        int_indexes = self.tokenizer.encode(text)
        gen_tokens = torch.tensor([int_indexes], dtype=torch.long)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                gen_tokens = gen_tokens.to(torch.long)
                next_token_logits = self.forward(gen_tokens)[:, -1, :]

                # Random sampling of token instead of most likely one to prevent repetition
                next_token = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
                
                gen_tokens = torch.cat((gen_tokens, next_token), dim=1)

        return self.tokenizer.decode(gen_tokens[0].tolist())