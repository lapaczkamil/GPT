import torch
import torch.nn as nn

class MaskedSelfAttention(nn.Module):
  def __init__(self, embedding_dim, bias=True):
    super().__init__()
    self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
    self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
    self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

  def forward(self, x):
    B, T, embedding_dim = x.shape
    
    q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

    attention_scores = (q @ k.transpose(1, 2)) / embedding_dim ** 0.5
    
    matrix = torch.ones(T, T)
    mask = torch.tril(matrix)

    attention_scores = attention_scores.masked_fill(mask == 0, -float('inf'))
    attention_scores = torch.softmax(attention_scores, dim=2)

    out = attention_scores @ v

    return out

class GPT(nn.Module):
  def __init__(self, vocab_size, embedding_dim, max_seq_len):
    super().__init__()
    self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
    self.lm_head = nn.Linear(embedding_dim, vocab_size)
    self.attention = MaskedSelfAttention(embedding_dim)
    self.layer_norm = nn.LayerNorm(embedding_dim)
    self.ffn_linear = nn.Linear(embedding_dim, 4 * embedding_dim)
    self.ffn_gelu = nn.GELU()
    self.ffn_linear_comp = nn.Linear(4 * embedding_dim, embedding_dim)
    self.ffn = nn.Sequential(nn.Linear(embedding_dim, 4 * embedding_dim), nn.GELU(), nn.Linear(4 * embedding_dim, embedding_dim))

  def forward(self, idx):
    print(idx.shape)
    B, T = idx.shape

    tok_emb = self.token_embedding(idx) # (B, T, embedding_dim)
    pos = torch.arange(0, T, device=idx.device)
    pos_emb = self.position_embedding(pos) # (T, embedding_dim)
    print(f"lm_head: {self.lm_head}")

    x = tok_emb + pos_emb

    x += self.attention(self.layer_norm(x)) # residual connection

    x += self.ffn(self.layer_norm(x))

    x = 
    x = self.lm_head(self.layer_norm(x))

    return x