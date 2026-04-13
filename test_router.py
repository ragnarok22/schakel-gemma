"""Load a fine-tuned LoRA adapter and run inference on sample Spanish inputs."""

import argparse
import json
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

# Must match the system prompt used in the training data.
SYSTEM_PROMPT = (
    "Clasifica la intención del usuario. Responde solo JSON válido "
    "con una clave intent y un valor entre DOMOTICA, MUSICA o GENERAL."
)

# Sample inputs covering all three intents.
TEST_INPUTS = [
    # MUSICA
    "pon jazz relajante",
    "quiero escuchar rock",
    # DOMOTICA
    "enciende la luz del salón",
    "sube la temperatura a 22 grados",
    # GENERAL
    "qué hora es",
    "cuéntame un chiste",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test the fine-tuned intent router")
    p.add_argument(
        "--adapter-path",
        default="./router-lora-final",
        help="Path to saved LoRA adapter",
    )
    p.add_argument(
        "--model-id", default="google/gemma-4-E2B-it", help="Base model HuggingFace ID"
    )
    p.add_argument(
        "--qlora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load base model in 4-bit",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_id: str,
    adapter_path: str,
    use_qlora: bool,
) -> tuple[PeftModel, AutoTokenizer]:
    """Load the base model and attach the trained LoRA adapter.

    We intentionally do NOT call merge_and_unload() — merging LoRA weights
    into a 4-bit quantised model degrades quality because the adapter was
    trained against dequantised views, not the raw 4-bit values.
    """
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    load_kwargs: dict = {"device_map": "auto"}
    if use_qlora:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def run_inference(model: PeftModel, tokenizer: AutoTokenizer, user_input: str) -> str:
    """Build a chat prompt, generate, and return only the new tokens."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    # Decode only the newly generated tokens.
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_intent(raw: str) -> str | None:
    """Try to extract the intent value from the model output."""
    try:
        data = json.loads(raw)
        return data.get("intent")
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        level=logging.INFO,
    )
    args = parse_args()

    logger.info("Loading base model: %s", args.model_id)
    logger.info("Loading adapter from: %s", args.adapter_path)
    model, tokenizer = load_model(args.model_id, args.adapter_path, args.qlora)

    print("\n" + "=" * 60)
    print("Schakel Intent Router — Inference Test")
    print("=" * 60)

    for user_input in TEST_INPUTS:
        raw = run_inference(model, tokenizer, user_input)
        intent = parse_intent(raw)

        print(f"\n  Input:  {user_input}")
        print(f"  Raw:    {raw}")
        if intent is not None:
            print(f"  Intent: {intent}")
        else:
            print("  Intent: PARSE FAILED")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
