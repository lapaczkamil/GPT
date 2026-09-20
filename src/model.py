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
    
    mask = torch.tril(torch.ones(T, T, device=x.device))

    attention_scores = attention_scores.masked_fill(mask == 0, -float('inf'))
    attention_scores = torch.softmax(attention_scores, dim=2)

    out = attention_scores @ v

    return out

class TransformerBlock(nn.Module):
  def __init__(self, embedding_dim):
    super().__init__()
    self.attention = MaskedSelfAttention(embedding_dim)
    self.ln_1 = nn.LayerNorm(embedding_dim)
    self.ln_2 = nn.LayerNorm(embedding_dim)
    self.ffn = nn.Sequential(nn.Linear(embedding_dim, 4 * embedding_dim), nn.GELU(), nn.Linear(4 * embedding_dim, embedding_dim))

  def forward(self, x):
    x = x + self.attention(self.ln_1(x))
    x = x + self.ffn(self.ln_2(x))
    
    return x

class GPT(nn.Module):
  def __init__(self, vocab_size, embedding_dim, max_seq_len):
    super().__init__()
    self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
    self.lm_head = nn.Linear(embedding_dim, vocab_size)
    self.final_ln = nn.LayerNorm(embedding_dim)
    self.blocks = nn.Sequential(*[TransformerBlock(embedding_dim) for _ in range(4)])


  def forward(self, idx):
    B, T = idx.shape

    tok_emb = self.token_embedding(idx) # (B, T, embedding_dim)
    pos = torch.arange(0, T, device=idx.device)
    pos_emb = self.position_embedding(pos) # (T, embedding_dim)

    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.lm_head(self.final_ln(x))

    return x
