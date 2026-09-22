# GPT from Scratch

Small Generative Pre-trained Transformer (GPT) in PyTorch

![GPT Architecture (decoder-only)](https://media.geeksforgeeks.org/wp-content/uploads/20260324090208535998/Decoder-only.webp)

## What's in the project

| File | Role |
|------|------|
| `src/model.py` | Embeddings, Transformer blocks (masked self-attention + FFN), generation |
| `src/dataset.py` | Text tokenization (tiktoken GPT-2) → `.bin` file, memmap-backed dataset |
| `src/train.py` | Training loop, checkpoints, generation samples |
| `src/generate.py` | Load a checkpoint and generate text |
| `src/config.py` | Configuration from `.env` |
| `data/raw/` | Raw text and tokenized `.bin` files |
| `checkpoints/` | Saved weights |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put a text file in `data/raw/` (e.g. `pan-tadeusz.txt`) and set the output token file name in `.env` (`TOKEN_FILE`).

## Configuration (`.env`)

Main variables:

- `TOKEN_FILE` - name of the token `.bin` file (e.g. `pan_tadeusz.bin`)
- `MAX_SEQ_LEN` - context length
- `EMBEDDING_DIM`, `BLOCK_NUM` - model size
- `VOCAB_SIZE` - `50257` for tiktoken GPT-2
- `batch_size`, `learning_rate`, `num_epochs`

For a small corpus (e.g. a single book), a sensible starting point is `MAX_SEQ_LEN=256`, `EMBEDDING_DIM=256`, `BLOCK_NUM=4–6`.

## Training

From the project root:

```bash
python3 src/train.py
```

If `data/raw/$TOKEN_FILE` is missing, the script tokenizes the data first. Every 5000 steps it saves a checkpoint to `checkpoints/`.

## Generation

Set the checkpoint path in `src/generate.py`, then:

```bash
cd src && python3 generate.py
```

Training checkpoints are saved as a dict - load weights with `ckpt["model"]`.

## Notes

- Tokenization uses **tiktoken** (`gpt2`); tokens are stored on disk as `uint16`.
- The dataset builds next-token-prediction windows via memmap (saves RAM).
