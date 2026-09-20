import torch
import torch.nn as nn

class MaskedAttention(nn.Module):
  def __init__(self, embedding_dim):
    super().__init__()

class GPT(nn.Module):
  def __init__(self, vocab_size, embedding_dim, max_seq_len):
    super().__init__()
    self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
    self.lm_head = nn.Linear(embedding_dim, vocab_size)

  def forward(self, idx):
    print(idx.shape)
    B, T = idx.shape

    tok_emb = self.token_embedding(idx) # (B, T, embedding_dim)
    pos = torch.arange(0, T, device=idx.device)
    pos_emb = self.position_embedding(pos) # (T, embedding_dim)
    print(f"lm_head: {self.lm_head}")
    x = self.lm_head(tok_emb + pos_emb)
    print(x.shape)
    return x