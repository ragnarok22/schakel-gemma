"""Fine-tune a causal LM as a Spanish intent router using QLoRA / LoRA + SFTTrainer."""

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a causal LM for intent routing")

    # Model / data paths
    p.add_argument(
        "--model-id", default="google/gemma-4-E2B-it", help="HuggingFace model ID"
    )
    p.add_argument("--output-dir", default="./router-lora", help="Checkpoint directory")
    p.add_argument(
        "--data-dir",
        default="./data",
        help="Directory with train.jsonl and valid.jsonl",
    )

    # Training hyper-parameters
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument(
        "--batch-size", type=int, default=4, help="Per-device train batch size"
    )
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max sequence length (keep small for intent routing)",
    )

    # LoRA hyper-parameters
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")

    # Quantisation
    p.add_argument(
        "--qlora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 4-bit QLoRA (default: on)",
    )

    # Logging / checkpointing cadence
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=50)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        level=logging.INFO,
    )


def load_data(data_dir: str) -> dict:
    """Load train / validation splits from JSONL files.

    Returns a HuggingFace DatasetDict.  If valid.jsonl is missing or empty the
    validation split is set to None so that evaluation is skipped.
    """
    data_path = Path(data_dir)
    train_path = data_path / "train.jsonl"
    valid_path = data_path / "valid.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if train_path.stat().st_size == 0:
        raise ValueError(f"Training file is empty: {train_path}")

    # Validation is optional — if the file is missing or empty we skip eval.
    has_valid = valid_path.exists() and valid_path.stat().st_size > 0

    if not has_valid:
        logger.warning("Validation file missing or empty — evaluation will be skipped")
        ds = load_dataset("json", data_files={"train": str(train_path)})
        return ds["train"], None

    ds = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(valid_path)},
    )
    return ds["train"], ds["validation"]


def build_model(
    model_id: str,
    use_qlora: bool,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model and tokenizer.

    When *use_qlora* is True the model is loaded in 4-bit NF4 quantisation via
    bitsandbytes.  Otherwise it is loaded in bf16.
    """
    if use_qlora and not torch.cuda.is_available():
        logger.warning(
            "QLoRA requested but CUDA is not available — "
            "bitsandbytes 4-bit quantisation requires a CUDA GPU. "
            "Consider passing --no-qlora for CPU / MPS."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    load_kwargs: dict = {"device_map": "auto"}
    if use_qlora:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    args = parse_args()

    logger.info("Model:  %s", args.model_id)
    logger.info("QLoRA:  %s", args.qlora)
    logger.info("Data:   %s", args.data_dir)
    logger.info("Output: %s", args.output_dir)

    # ---- data ----
    train_ds, valid_ds = load_data(args.data_dir)
    logger.info("Train samples: %d", len(train_ds))
    if valid_ds is not None:
        logger.info("Valid samples: %d", len(valid_ds))

    # ---- model ----
    model, tokenizer = build_model(args.model_id, args.qlora)

    # ---- LoRA ----
    # Target both attention and MLP projections for better classification.
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # ---- training config ----
    eval_strategy = "steps" if valid_ds is not None else "no"

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_strategy != "no" else None,
        save_steps=args.save_steps,
        bf16=True,
        max_length=args.max_length,
        report_to="none",
    )

    # ---- trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting training …")
    trainer.train()

    # ---- save ----
    final_dir = f"{args.output_dir}-final"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Adapter and tokenizer saved to %s", final_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception:
        logger.exception("Training failed")
        raise
