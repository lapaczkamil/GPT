import torch
from model import GPT
from config import Config

def test_kv_cache_matches_naive():
  config = Config()

  torch.manual_seed(0)
  model = GPT(vocab_size=config.vocab_size, embedding_dim=config.embedding_dim, max_seq_len=config.max_seq_len, layer_num=config.layer_num, 
              num_heads=config.num_heads, dropout=config.dropout)

  model.eval()
  idx = torch.randint(0, 100, (1, 8))

  with torch.no_grad():
    out_kv = model.generate(idx.clone(), max_new_tokens=16, use_kv_cache=True)
    out_naive = model.generate(idx.clone(), max_new_tokens=16, use_kv_cache=False)
  
  assert torch.equal(out_kv, out_naive)