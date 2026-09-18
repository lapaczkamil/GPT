from typing import Any, cast

import tiktoken
import torch
from datasets import Dataset, load_dataset

CACHE_DIR = "/home/kamil/Projects/GPT/data/raw"
MAX_SEQ_LEN = 256


class StoryDataset(torch.utils.data.Dataset):
  def __init__(self, all_tokens, max_seq_len):
    self.all_tokens = all_tokens
    self.max_seq_len = max_seq_len
  
  def __len__(self):
    return self.all_tokens.shape[0] - self.max_seq_len

  def __getitem__(self, index):
    sentance = self.all_tokens[index : index + self.max_seq_len + 1]
    return sentance[:self.max_seq_len], sentance[1:]

def prepare_data():
  dataset = cast(
      Dataset,
      load_dataset("roneneldan/TinyStories", split="train", cache_dir=CACHE_DIR),
  )
  encoder = tiktoken.get_encoding("gpt2")
  all_tokens = []

  for article in dataset:
    row = cast(dict[str, Any], article)
    tokens = encoder.encode(row["text"])
    all_tokens.extend(tokens)
  
  return all_tokens

if __name__ == "__main__":
  tokens = prepare_data()
  my_dataset = StoryDataset(all_tokens=tokens, max_seq_len=MAX_SEQ_LEN)

  dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=4, shuffle=True)

  x, y = next(iter(dataloader))
  print(f"X: {x.shape}")
  print(f"Y: {y.shape}")