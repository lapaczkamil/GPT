import dataset
from model import GPT
from dotenv import load_dotenv
import os
import torch
import numpy as np
import tqdm
 
load_dotenv()

vocab_size = int(os.getenv("VOCAB_SIZE", "50257"))
embedding_dim = int(os.getenv("EMBEDDING_DIM", "64"))
max_seq_len = int(os.getenv("MAX_SEQ_LEN", "256"))

CACHE_DIR = os.getenv("CACHE_DIR", "./data/raw")
PROCESSED_FILE_PATH = os.path.join(CACHE_DIR, "tinystories_tokens.bin")

gpt_model = GPT(vocab_size=vocab_size, embedding_dim=embedding_dim, max_seq_len=max_seq_len)

learning_rate = 1e-4
weight_decay = 1e-2
num_epochs = 10

optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_func = torch.nn.CrossEntropyLoss()

if torch.cuda.is_available():
  device = 'cuda'
elif torch.backends.mps.is_available():
  device = 'mps'
else:
  device = 'cpu'

print(f'Training using: {device}')

gpt_model.to(device)

if not os.path.exists(PROCESSED_FILE_PATH):
  dataset.prepare_data()

memmap_test = np.memmap(PROCESSED_FILE_PATH, dtype=np.uint16, mode='r')
print(f"Test wczytania: plik widziany z dysku ma {len(memmap_test):,} tokenów.")

my_dataset = dataset.StoryDataset(processed_file_path=PROCESSED_FILE_PATH, max_seq_len=max_seq_len)

dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=4, shuffle=False) # Do zoptymalizowania w przyszlosci zeby bylo shuffle=True

for epoch in range(num_epochs):
  print(f"Epoch: {epoch} / {num_epochs}")
  for i, (x, y) in enumerate(dataloader):
    x = x.to(device)
    y = y.to(device)

    logits = gpt_model(x)
    logits = logits.reshape(-1, logits.shape[2])
    y = y.reshape(-1)

    loss = loss_func(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
      print(f"Score {i} / {len(dataloader)}: {loss.item()}")

print("Training finished. Saving weights...")
torch.save(gpt_model.state_dict(), "gpt_model.pth")
print("Done")
  