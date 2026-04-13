# Schakel Intent Router

QLoRA fine-tuning of a small instruction-tuned LLM (default: Google Gemma 4 E2B) as a Spanish-language intent classifier.

Classifies user input into one of three intents: **DOMOTICA**, **MUSICA**, **GENERAL**.
The model learns to output strict JSON: `{"intent":"DOMOTICA"}`.

## Setup

```bash
uv sync
```

## Prepare data

Place training data in `data/`:

- `data/train.jsonl`
- `data/valid.jsonl`

Each line is a JSON object with a `messages` array (system, user, assistant) in the standard chat format.

## Train

```bash
uv run python train_router.py
```

Common overrides:

```bash
# Custom model, more epochs, lower learning rate
uv run python train_router.py --model-id google/gemma-3-1b-it --num-epochs 5 --lr 1e-4

# Disable QLoRA (plain LoRA in bf16, no CUDA required)
uv run python train_router.py --no-qlora
```

Run `uv run python train_router.py --help` for all options.

## Test

After training completes, test the fine-tuned adapter:

```bash
uv run python test_router.py
```

With a custom adapter or model:

```bash
uv run python test_router.py --adapter-path ./router-lora-final --model-id google/gemma-3-1b-it
```
