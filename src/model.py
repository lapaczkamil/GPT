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

class MultiHeadAttention(nn.Module):
  def __init__(self, embedding_dim, num_heads, dropout=0.0, bias=True):
    super().__init__()
    self.num_heads = num_heads
    self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
    self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
    self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
    self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
    self.attention_dropout = nn.Dropout(dropout)

  def forward(self, x):
    batch_size, seq_len, embedding_dim = x.shape
    head_dim = embedding_dim // self.num_heads
  
    q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

    q = q.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

    attention_scores = (q @ k.transpose(2, 3)) / head_dim ** 0.5
    casual_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

    attention_scores = attention_scores.masked_fill(casual_mask == 0, -float('inf'))
    attention_scores = torch.softmax(attention_scores, dim=-1)
    attention_scores = self.attention_dropout(attention_scores)

    y = attention_scores @ v

    y = y.transpose(1, 2).contiguous().reshape(batch_size, seq_len, embedding_dim)
    return self.out_proj(y)

class TransformerBlock(nn.Module):
  def __init__(self, embedding_dim, head_num, dropout=0.0):
    super().__init__()
    self.attention = MultiHeadAttention(embedding_dim, head_num, dropout)
    self.ln_1 = nn.LayerNorm(embedding_dim)
    self.ln_2 = nn.LayerNorm(embedding_dim)
    self.ffn = nn.Sequential(nn.Linear(embedding_dim, 4 * embedding_dim), nn.GELU(), nn.Linear(4 * embedding_dim, embedding_dim))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = x + self.dropout(self.attention(self.ln_1(x)))
    x = x + self.dropout(self.ffn(self.ln_2(x)))
    
    return x

class GPT(nn.Module):
  def __init__(self, vocab_size, embedding_dim, max_seq_len, layer_num, num_heads, dropout=0.0):
    super().__init__()
    self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
    self.lm_head = nn.Linear(embedding_dim, vocab_size)
    self.final_ln = nn.LayerNorm(embedding_dim)
    self.layers = nn.Sequential(*[TransformerBlock(embedding_dim, num_heads, dropout) for _ in range(layer_num)])
    self.max_seq_len = max_seq_len
    self.drop = nn.Dropout(dropout)

  def forward(self, idx):
    B, T = idx.shape

    tok_emb = self.token_embedding(idx) # (B, T, embedding_dim)
    pos = torch.arange(0, T, device=idx.device)
    pos_emb = self.position_embedding(pos) # (T, embedding_dim)

    x = self.drop(tok_emb + pos_emb)
    x = self.layers(x)
    x = self.lm_head(self.final_ln(x))

    return x

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.max_seq_len:]
      logits = self(idx_cond)
      logits = logits[:,-1,:]
      probs = torch.softmax(logits, dim=1)
      idx_next = torch.multinomial(probs, num_samples=1)

      idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
  

