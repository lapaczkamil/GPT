import torch
import torch.nn as nn

# class MaskedSelfAttention(nn.Module):
#   def __init__(self, embedding_dim, bias=True):
#     super().__init__()
#     self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
#     self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
#     self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

#   def forward(self, x):
#     B, T, embedding_dim = x.shape
    
#     q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

#     attention_scores = (q @ k.transpose(1, 2)) / embedding_dim ** 0.5
    
#     mask = torch.tril(torch.ones(T, T, device=x.device))

#     attention_scores = attention_scores.masked_fill(mask == 0, -float('inf'))
#     attention_scores = torch.softmax(attention_scores, dim=2)

#     out = attention_scores @ v

#     return out

class MultiHeadAttention(nn.Module):
  def __init__(self, embedding_dim, num_heads, dropout=0.0, bias=True):
    super().__init__()
    self.num_heads = num_heads
    self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
    self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
    self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
    self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
    self.attention_dropout = nn.Dropout(dropout)

  def forward(self, idx, kv_cache):
    batch_size, seq_len, embedding_dim = idx.shape


    head_dim = embedding_dim // self.num_heads
  
    q, k, v = self.q_proj(idx), self.k_proj(idx), self.v_proj(idx)

    q = q.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2) # (batch_size, num_heads, seq_len, head_dim)

    if kv_cache is not None:
      k = torch.cat([kv_cache['k'], k], dim=2)
      v = torch.cat([kv_cache['v'], v], dim=2)
      kv_cache['k'] = k
      kv_cache['v'] = v
    else:
      kv_cache = {"k": k, "v": v}

    # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
    attention_scores = (q @ k.transpose(2, 3)) / head_dim ** 0.5 

    if seq_len > 1 and :
      causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device))
      attention_scores = attention_scores.masked_fill(causal_mask == 0, -float('inf')) # Not needed with KV-cache

    attention_scores = torch.softmax(attention_scores, dim=-1)
    attention_scores = self.attention_dropout(attention_scores)

    y = attention_scores @ v

    y = y.transpose(1, 2).contiguous().reshape(batch_size, seq_len, embedding_dim)
    return self.out_proj(y), kv_cache

class TransformerBlock(nn.Module):
  def __init__(self, embedding_dim, head_num, dropout=0.0):
    super().__init__()
    self.attention = MultiHeadAttention(embedding_dim, head_num, dropout)
    self.ln_1 = nn.LayerNorm(embedding_dim)
    self.ln_2 = nn.LayerNorm(embedding_dim)
    self.ffn = nn.Sequential(nn.Linear(embedding_dim, 4 * embedding_dim), nn.GELU(), nn.Linear(4 * embedding_dim, embedding_dim))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, kv_cache=None):
    attn_out, kv_cache = self.attention(self.ln_1(x), kv_cache)
    x = x + self.dropout(attn_out)
    x = x + self.dropout(self.ffn(self.ln_2(x)))
    
    return x, kv_cache

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

  def forward(self, idx, kv_caches=None):
    B, T = idx.shape

    if kv_caches is None:
      kv_caches = [None] * len(self.layers)

    past_len = 0 if kv_caches[0] is None else kv_caches[0]["k"].size(2)
    
    tok_emb = self.token_embedding(idx) # (B, T, embedding_dim)
    pos = torch.arange(past_len, past_len + T, device=idx.device)

    pos_emb = self.position_embedding(pos) # (T, embedding_dim)

    x = self.drop(tok_emb + pos_emb)

    new_caches = []
    for i, layer in enumerate(self.layers):
      x, new_cache = layer(x, kv_cache=kv_caches[i])
      new_caches.append(new_cache)

    logits = self.lm_head(self.final_ln(x))

    return logits, new_caches

  def generate(self, idx, max_new_tokens, use_kv_cache=True):
    # prefill

    logits, caches = self(idx, kv_caches=None)
    idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    idx = torch.cat([idx, idx_next], dim=1)

    # decode
    for _ in range(max_new_tokens - 1):
      if use_kv_cache is True:
        logits, caches = self(idx_next, kv_caches=caches)
      else:
        logits, _ = self(idx[:, -self.max_seq_len:], kv_caches=None)

      probs = torch.softmax(logits[:, -1, :], dim=-1)
      idx_next = torch.argmax(probs, dim=-1, keepdim=True)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx
  

