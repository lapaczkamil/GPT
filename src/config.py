import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TrainingConfig:
  vocab_size: int = int(os.getenv("VOCAB_SIZE", "50257"))
  embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "64"))
  max_seq_len: int = int(os.getenv("MAX_SEQ_LEN", "256"))
  block_num: int = int(os.getenv("BLOCK_NUM", "6"))
  CHECKPOINT_DIR: str = str(os.getenv("CHECKPOINT_DIR"))
  CACHE_DIR: str = os.getenv("CACHE_DIR", "./data/raw")
  token_file = str(os.getenv("TOKEN_FILE"))
  PROCESSED_FILE_PATH = os.path.join(CACHE_DIR, token_file)

  batch_size: int = 6
  learning_rate: float = 3e-4
  weight_decay: float = 0.1
  num_epochs: int = 1