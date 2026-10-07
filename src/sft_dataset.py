from typing import Any, cast
import tiktoken
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
import os
from tqdm import tqdm
import numpy as np
from config import DatasetConfig



class StorySftDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, encoder, max_seq_len):
    self.data = load_dataset(dataset, split="train")
    self.encoder = encoder
    self.max_seq_len = max_seq_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    row = self.data[index]
    query = "[QUERY]" + str(row['query'])
    answer = "[ANSWER]" + str(row['answer'])

    query_encoded = self.encoder.encode(query)
    query_len = len(query_encoded)
    
    ids = self.encoder.encode(query + answer)
    ids = ids[: self.max_seq_len + 1]

    pad_id = 0
    if len(ids) < self.max_seq_len + 1:
      ids = ids + [pad_id] * (self.max_seq_len + 1 - len(ids))

    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
  
    y[:query_len - 1] = -100
    y[y == pad_id] = -100

    return x, y



def main():
  config = DatasetConfig()

  encoder = tiktoken.get_encoding("gpt2")

  my_dataset = StorySftDataset(config.dataset_sft, encoder, config.max_seq_len)

  dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=4, shuffle=True) # Do zoptymalizowania w przyszlosci zeby bylo shuffle=True

  x, y = next(iter(dataloader))
  print(f"X: {x.shape}")
  print(f"Y: {y.shape}")

if __name__ == "__main__":
  main()
