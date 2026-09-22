import torch
from model import GPT
from dotenv import load_dotenv
import os
import tiktoken

load_dotenv()

CHECKPOINTS_DIR = os.path.join(os.getcwd(), "checkpoints/")

embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
max_seq_len = int(os.getenv("MAX_SEQ_LEN", "1024"))
num_embeddings = int(os.getenv("VOCAB_SIZE", "50257"))
block_num = int(os.getenv("BLOCK_NUM", "6"))

model = GPT(vocab_size=num_embeddings, embedding_dim=embedding_dim, max_seq_len=max_seq_len, block_num=block_num)

ckpt = torch.load(os.path.join(CHECKPOINTS_DIR, 'gpt_step_20000_pan_tedeusz.pth'), map_location='cpu', weights_only=False)
model.load_state_dict(ckpt["model"])

out = model.generate(idx=torch.zeros(1, 1, dtype=torch.long), max_new_tokens=100)

encoding = tiktoken.encoding_for_model("gpt2")

print(encoding.decode(out[0].tolist()))