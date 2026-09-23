from typing import Any, cast
import tiktoken
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
import os
from tqdm import tqdm
import numpy as np
from config import DatasetConfig



class StoryDataset(torch.utils.data.Dataset):
  def __init__(self, processed_file_path, max_seq_len):
    self.all_tokens = np.memmap(processed_file_path, dtype=np.uint16, mode='r')
    self.max_seq_len = max_seq_len

  def __len__(self):
    return self.all_tokens.shape[0] - self.max_seq_len

  def __getitem__(self, index):
    sentance = self.all_tokens[index : index + self.max_seq_len + 1]
    x = sentance[:self.max_seq_len]
    y = sentance[1:]

    x_tensor = torch.tensor(x.astype(np.int64), dtype=torch.long)
    y_tensor = torch.tensor(y.astype(np.int64), dtype=torch.long)

    return x_tensor, y_tensor 



def prepare_data(config):
  dataset = load_dataset(
      config.dataset,
      split="train",
      cache_dir=config.CACHE_DIR,
  )
  encoder = tiktoken.get_encoding("gpt2")
  batch_tokens = []

  total = 0
  with open(config.PROCESSED_FILE_PATH, 'wb') as f:
    for i, article in enumerate(tqdm(dataset, total=len(dataset), desc="Tokenizing")):
      row = cast(dict[str, Any], article)
      tokens = encoder.encode(row["text"])

      if total + len(tokens) > config.MAX_TOKENS:
        tokens = tokens[: config.MAX_TOKENS - total]
        
      batch_tokens.extend(tokens)
      total += len(tokens)

      if total >= config.MAX_TOKENS:
        break

      if (i + 1) % 10000 == 0:
        batch_array = np.array(batch_tokens, dtype=np.uint16)
        batch_array.tofile(f)
        batch_tokens=[]

    print(len(batch_tokens))

    if len(batch_tokens) > 0:
        batch_array = np.array(batch_tokens, dtype=np.uint16)
        batch_array.tofile(f)
        batch_tokens=[]
  print("Done")


def main():
  config = DatasetConfig()

  if not os.path.exists(config.PROCESSED_FILE_PATH):
    prepare_data(config)

  memmap_test = np.memmap(config.PROCESSED_FILE_PATH, dtype=np.uint16, mode='r')
  print(f"Test wczytania: plik widziany z dysku ma {len(memmap_test):,} tokenów.")

  my_dataset = StoryDataset(processed_file_path=config.PROCESSED_FILE_PATH, max_seq_len=config.max_seq_len)

  dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=4, shuffle=True) # Do zoptymalizowania w przyszlosci zeby bylo shuffle=True

  x, y = next(iter(dataloader))
  print(f"X: {x.shape}")
  print(f"Y: {y.shape}")

if __name__ == "__main__":
  main()
