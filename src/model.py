import torch
import torch.nn as nn

class GPT(nn.Module):
  def __init__(self, vocab_size, embedding_dim, max_seq_len):
    super().__init__()
    self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, idx):
      B, T = idx.shape

      tok_emb = self.token_embedding(idx) # (B, T, embedding_dim)

      pos = torch.arange(0, T, device=idx.device)

      pos_emb = self.postion_embedding(pos) # (T, embedding_dim)

      x = tok_emb + pos_emb

      return x