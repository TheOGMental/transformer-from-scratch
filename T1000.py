from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

@dataclass
class GPTConfig:
	# default test values -- too small for a real language model, but big enough for testing
	d_vocab: int = 10_000
	d_model: int = 128
	d_mlp: int = 512
	n_heads: int = 4
	d_head: int = 32
	n_layers: int = 6
	act_fn: type[nn.Module] = nn.ReLU

	@property
	def n_params(self) -> int:
		"an estimate of the number of parameters"
		return (
			self.d_vocab * self.d_model # embeddings (and tied unembeddings)
			+ (
				self.d_model * self.d_mlp * 2 # mlp weights
				+ self.d_model + self.d_mlp # mlp bias
				+ self.n_heads * ( # number of heads
					4 * self.d_model * self.d_head # 4 because Q, K, O, V
				)
			) * self.n_layers, # for each layer
		)

# note: the residual stream is `n_context` by `d_model`

# this is the row-wise (last dimension) softmax of x
# F.softmax(x, dim=-1)

class AttentionHead(nn.Module):
	
	def __init__(self, cfg: GPTConfig):
		super().__init__()
		self.W_q = torch.nn.Parameter(torch.rand(cfg.d_model, cfg.d_head))
		self.W_kT = torch.nn.Parameter(torch.rand(cfg.d_head, cfg.d_model))
		self.W_o = torch.nn.Parameter(torch.rand(cfg.d_model, cfg.d_head))
		self.W_vT = torch.nn.Parameter(torch.rand(cfg.d_head, cfg.d_model))
	
	def masking_matrix (self, n_context: Int) -> Float[torch.Tensor, "n_context n_context"]:
		m = torch.full((n_context, n_context), -torch.inf)
		return torch.triu(m, diagonal=1)

	def forward(self, x: Int[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
		# Compute attention pattern
		pattern = x @ self.W_q @ self.W_kT @ x.T
		pattern += self.masking_matrix(x.size()[0])
		pattern = torch.nn.functional.softmax(pattern, dim=1) @ x @ self.W_o @ self.W_vT

		return pattern

# List of heads needs to be of nn.Module type 

class MultiHeadedAttention(nn.Module):

	def __init__(self, cfg: GPTConfig):
		super().__init__()
		self.head_list = nn.ModuleList([AttentionHead(cfg) for _ in range(cfg.n_heads)])

	def forward(self, x: Int[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
		output = x
		for h in self.head_list:
			output += h.forward(x)
		return output


class MLP(nn.Module):

	def __init__(self, cfg: GPTConfig):
		super().__init__()
		self.W_m_down = torch.nn.Parameter(torch.rand(cfg.d_model, cfg.d_mlp))
		self.W_m_up = torch.nn.Parameter(torch.rand(cfg.d_mlp, cfg.d_model))
		self.B = torch.nn.Parameter(torch.rand(cfg.d_mlp, 1))
		self.act_function = cfg.act_fn()

	def forward(self, x: Int[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
		''' 
		Instead of looping over each column to add B, make matrix of size (d_mlp, n_context) where
		its columns correspond to the values of B
		'''
		B_matrix = self.B @ torch.full((1, x.shape[0]), 1.0)

		x_out = (self.W_m_up @ x.T) + B_matrix
		x_out = ((self.W_m_down @ self.act_function(x_out)).T) + x
		
		return x_out
	
class Transformer(nn.Module):

	def __init__(self, cfg: GPTConfig):
		super().__init__()
		self.embedding = nn.Embedding(cfg.d_vocab, cfg.d_model)
		self.MHA_layers = nn.ModuleList([MultiHeadedAttention(cfg) for _ in range(cfg.n_layers)])
		self.MLP_layers = nn.ModuleList([MLP(cfg) for _ in range(cfg.n_layers)])
		# Use nn.Embedding for the embedding, and CAN new linear layer OR use transpose for the unembedding
		self.out_layer = nn.Linear(cfg.d_model, cfg.d_vocab)

	def forward(self, x: Int[torch.Tensor, "n_context"]) -> Float[torch.Tensor, "n_context d_vocab"]:
		X = self.embedding(x)
		for MHA, MLP in zip(self.MHA_layers, self.MLP_layers):
			X = MHA.forward(X)
			X = MLP.forward(X)
		return self.out_layer(X)