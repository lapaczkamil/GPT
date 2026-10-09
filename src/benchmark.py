import time
import torch
from model import GPT
import os
import tiktoken
from config import Config

config = Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT(vocab_size=config.vocab_size, embedding_dim=config.embedding_dim, max_seq_len=config.max_seq_len, layer_num=config.layer_num, num_heads=config.num_heads, dropout=config.dropout)

ckpt = torch.load("gpt_model_sft.pth", map_location='cpu', weights_only=True)
model.load_state_dict(ckpt)
model.to(device)
model.eval()

tokenizer = tiktoken.encoding_for_model("gpt2")
prompt_ids = tokenizer.encode("[QUERY] Make a story about cats. [ANSWER]")
idx = torch.tensor([prompt_ids], device=device)
max_new = 200
warmup = 3
runs = 10


for _ in range(warmup):
  with torch.no_grad():
    model.generate(idx.clone(), max_new_tokens=max_new)

torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(runs):
  with torch.no_grad():
    model.generate(idx.clone(), max_new_tokens=max_new, use_kv_cache=False)
  torch.cuda.synchronize()
t1 = time.perf_counter()

elapsed = (t1 - t0) / runs
tok_s = max_new / elapsed
print(f"naive: {elapsed:.3f}s, {tok_s:.1f} tok/s")


t0 = time.perf_counter()
for _ in range(runs):
  with torch.no_grad():
    model.generate(idx.clone(), max_new_tokens=max_new, use_kv_cache=True)
  torch.cuda.synchronize()
t1 = time.perf_counter()

elapsed = (t1 - t0) / runs
tok_s = max_new / elapsed
print(f"KV-cache: {elapsed:.3f}s, {tok_s:.1f} tok/s")