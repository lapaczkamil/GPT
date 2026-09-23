import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
  vocab_size: int = int(os.getenv("VOCAB_SIZE", "50257"))
  embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "64"))
  max_seq_len: int = int(os.getenv("MAX_SEQ_LEN", "256"))
  layer_num: int = int(os.getenv("LAYER_NUM", "6"))
  dropout: float = float(os.getenv("DROPOUT", "0.1"))
  CHECKPOINT_DIR: str = str(os.getenv("CHECKPOINT_DIR"))
  CACHE_DIR: str = os.getenv("CACHE_DIR", "./data/processed")
  token_file = str(os.getenv("TOKEN_FILE"))
  PROCESSED_FILE_PATH = os.path.join(CACHE_DIR, token_file)


@dataclass
class DatasetConfig(Config):
  dataset: str = str(os.getenv("DATASET"))
  MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1_000_000"))

@dataclass
class TrainingConfig(Config):
  batch_size: int = int(os.getenv("BATCH_SIZE", "6"))
  learning_rate: float = float(os.getenv("LEARNING_RATE", "3e-4"))
  weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.1"))
  num_epochs: int = int(os.getenv("NUM_EPOCHS", "1"))

