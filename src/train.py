import dataset
from model import GPT
from dotenv import load_dotenv
import os
import torch
import numpy as np
import tiktoken
from config import TrainingConfig

class Trainer:
  def __init__(self, model, optimizer, loss_func, dataloader, device, num_epochs, tokenizer, checkpoint_dir):
    self.model = model
    self.optimizer = optimizer
    self.loss_func = loss_func
    self.dataloader = dataloader
    self.num_epochs = num_epochs
    self.device = device
    self.tokenizer = tokenizer
    self.checkpoint_dir = checkpoint_dir

  def train(self):
    for epoch in range(self.num_epochs):
      print(f"Epoch: {epoch} / {self.num_epochs}")
      for i, (x, y) in enumerate(self.dataloader):
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        logits = logits.reshape(-1, logits.shape[2])
        y = y.reshape(-1)

        loss = self.loss_func(logits, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (i + 1) % 100 == 0:
          print(f"| {i} / {len(self.dataloader)} |  Loss: {loss.item()}")
        
        if (i + 1) % 5000 == 0:
          self.generate_sample()
          self.save_checkpoint(self.checkpoint_dir, i, loss)

    print("Training finished. Saving weights...")
    torch.save(self.model.state_dict(), "gpt_model.pth")
    print("Done")

  def save_checkpoint(self, checkpoint_path, step, loss):
    path = os.path.join(checkpoint_path, f"gpt_step_{step + 1}_pan_tedeusz.pth")
    torch.save(
        {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step + 1,
            "loss": loss.item(),
        },
        path,
    )
    print(f"Saved {path}")

  def generate_sample(self):
    tensor = torch.zeros(1,1, dtype=torch.long).to(self.device)
    logits = self.model.generate(idx=tensor, max_new_tokens=1024)
    text = self.tokenizer.decode(logits[0].tolist())
    print(text)

def main():
  config = TrainingConfig()

  os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

  model = GPT(vocab_size=config.vocab_size, embedding_dim=config.embedding_dim, max_seq_len=config.max_seq_len, block_num=config.block_num)
  optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
  loss_func = torch.nn.CrossEntropyLoss()

  if torch.cuda.is_available():
    device = 'cuda'
  elif torch.backends.mps.is_available():
    device = 'mps'
  else:
    device = 'cpu'

  print(f'Training using: {device}')
  model.to(device)
  
  if not os.path.exists(config.PROCESSED_FILE_PATH):
    dataset.prepare_data()

  memmap_test = np.memmap(config.PROCESSED_FILE_PATH, dtype=np.uint16, mode='r')
  print(f"Test wczytania: plik widziany z dysku ma {len(memmap_test):,} tokenów.")

  my_dataset = dataset.StoryDataset(processed_file_path=config.PROCESSED_FILE_PATH, max_seq_len=config.max_seq_len)
  dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=config.batch_size, shuffle=True) # Do zoptymalizowania w przyszlosci zeby bylo shuffle=True

  trainer = Trainer(model=model, 
                    optimizer=optimizer, 
                    loss_func=loss_func, 
                    dataloader=dataloader, 
                    device=device, 
                    num_epochs=config.num_epochs, 
                    tokenizer=tiktoken.encoding_for_model("gpt2"), 
                    checkpoint_dir=config.CHECKPOINT_DIR)
  trainer.train()


if __name__ == "__main__":
  main()