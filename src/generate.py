import torch
from model import GPT
from dotenv import load_dotenv
import os
import tiktoken
from config import Config

config = Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT(vocab_size=config.vocab_size, embedding_dim=config.embedding_dim, max_seq_len=config.max_seq_len, layer_num=config.layer_num, num_heads=config.num_heads, dropout=config.dropout)

ckpt = torch.load("gpt_model_converted.pth", map_location='cpu', weights_only=True)
model.load_state_dict(ckpt)
model.to(device)
model.eval()

prompt = "My favorite animal is"
tokenizer = tiktoken.encoding_for_model("gpt2")

token_ids = tokenizer.encode(prompt)
tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
out = model.generate(idx=tensor, max_new_tokens=100)


print(tokenizer.decode(out[0].tolist()))