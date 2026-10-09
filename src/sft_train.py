import sft_dataset
from model import GPT
from dotenv import load_dotenv
import os
import torch
import numpy as np
import tiktoken
from config import TrainingConfig

class Trainer:
  def __init__(self, model, optimizer, loss_func, dataloader, device, num_epochs, tokenizer, checkpoint_dir, token_file):
    self.model = model
    self.optimizer = optimizer
    self.loss_func = loss_func
    self.dataloader = dataloader
    self.num_epochs = num_epochs
    self.device = device
    self.tokenizer = tokenizer
    self.checkpoint_dir = checkpoint_dir
    self.token_file = token_file

  def train(self):
    for epoch in range(self.num_epochs):
      print(f"Epoch: {epoch + 1} / {self.num_epochs}")
      for i, (x, y) in enumerate(self.dataloader):
        x = x.to(self.device)
        y = y.to(self.device)

        logits, _ = self.model(x)
        logits = logits.reshape(-1, logits.shape[2])
        y = y.reshape(-1)

        loss = self.loss_func(logits, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (i + 1) % 100 == 0:
          print(f"| {i + 1} / {len(self.dataloader)} |  Loss: {loss.item()}")
        
        if (i + 1) % 5000 == 0:
          self.generate_sample()
          self.save_checkpoint(self.checkpoint_dir, i, loss)

    print("Training finished. Saving weights...")
    torch.save(self.model.state_dict(), "gpt_model_sft.pth")
    print("Done")

  def save_checkpoint(self, checkpoint_path, step, loss):
    path = os.path.join(checkpoint_path, f"gpt_step_{step + 1}_{self.token_file.split('.')[0]}.pth")
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
    self.model.eval()
    prompt = "[QUERY] Write a short story about a cat. [ANSWER] "
    ids = self.tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=self.device)

    with torch.no_grad():
      out = self.model.generate(idx, max_new_tokens=100)

    print(self.tokenizer.decode(out[0].tolist()))
    self.model.train()

def main():
  config = TrainingConfig()

  os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

  model = GPT(vocab_size=config.vocab_size, embedding_dim=config.embedding_dim, max_seq_len=config.max_seq_len, layer_num=config.layer_num, dropout=config.dropout, num_heads=config.num_heads)
  optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
  loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
  encoder = tiktoken.get_encoding("gpt2")

  if torch.cuda.is_available():
    device = 'cuda'
  elif torch.backends.mps.is_available():
    device = 'mps'
  else:
    device = 'cpu'

  print(f'Training using: {device}')
  model.to(device)

  ckpt = torch.load("gpt_model_converted.pth", map_location=device, weights_only=True)
  model.load_state_dict(ckpt)
  
  my_dataset = sft_dataset.StorySftDataset(dataset=config.dataset_sft, encoder=encoder, max_seq_len=config.max_seq_len)
  dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=config.batch_size, shuffle=True)

  trainer = Trainer(model=model, 
                    optimizer=optimizer, 
                    loss_func=loss_func, 
                    dataloader=dataloader, 
                    device=device, 
                    num_epochs=config.num_epochs, 
                    tokenizer=tiktoken.encoding_for_model("gpt2"), 
                    checkpoint_dir=config.CHECKPOINT_DIR,
                    token_file=config.token_file
                    )
  trainer.train()


if __name__ == "__main__":
  main()