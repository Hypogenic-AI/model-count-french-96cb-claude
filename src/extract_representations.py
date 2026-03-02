"""
Extract hidden-state representations from Llama 3 8B for numbers 0-999
in four formats: digit strings, English words, France French, Belgian French.

Saves representations per layer to disk for downstream probing.
"""

import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

WORKSPACE = "/workspaces/model-count-french-96cb-claude"
DATA_PATH = os.path.join(WORKSPACE, "datasets/french_numbers/french_numbers.json")
OUTPUT_DIR = os.path.join(WORKSPACE, "results/representations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# ── English number words ──────────────────────────────────────────────────

ENGLISH_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
ENGLISH_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
]

def english_number(n: int) -> str:
    if n < 0 or n > 999:
        raise ValueError(n)
    if n < 20:
        return ENGLISH_ONES[n]
    if n < 100:
        t, u = divmod(n, 10)
        return ENGLISH_TENS[t] if u == 0 else f"{ENGLISH_TENS[t]}-{ENGLISH_ONES[u]}"
    h, r = divmod(n, 100)
    base = f"{ENGLISH_ONES[h]} hundred"
    if r == 0:
        return base
    return f"{base} {english_number(r)}"


def load_french_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_prompts(data):
    """Build prompt strings for all 4 formats for each number."""
    prompts = {"digits": [], "english": [], "french": [], "belgian": []}
    for entry in data:
        n = entry["number"]
        prompts["digits"].append(f"The number is {n}.")
        prompts["english"].append(f"The number is {english_number(n)}.")
        prompts["french"].append(f"Le nombre est {entry['french']}.")
        # Belgian: use french_belgian if different, otherwise same as french
        belgian_word = entry["french_belgian"] if entry["french_belgian"] else entry["french"]
        prompts["belgian"].append(f"Le nombre est {belgian_word}.")
    return prompts


def extract_last_token_hidden_states(model, tokenizer, texts, batch_size=32, device="cuda:0"):
    """
    Extract hidden states at the LAST token position for each text.
    Returns: dict mapping layer_idx -> np.array of shape (num_texts, hidden_dim)
    """
    model.eval()
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size

    all_hidden = {layer: np.zeros((len(texts), hidden_dim), dtype=np.float32)
                  for layer in range(num_layers)}

    for start in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)

        # Find last non-padding token for each example
        attention_mask = inputs["attention_mask"]  # (batch, seq_len)
        # Last non-padding position: sum of mask - 1
        last_positions = attention_mask.sum(dim=1) - 1  # (batch,)

        for layer_idx in range(num_layers):
            hs = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
            for i in range(len(batch_texts)):
                pos = last_positions[i].item()
                all_hidden[layer_idx][start + i] = hs[i, pos].cpu().numpy()

        # Free memory
        del outputs, hidden_states
        torch.cuda.empty_cache()

    return all_hidden


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        output_hidden_states=True,
    )
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}, Hidden dim: {model.config.hidden_size}")

    # Load data and build prompts
    data = load_french_data()
    prompts = build_prompts(data)

    # Save prompts for reproducibility
    with open(os.path.join(OUTPUT_DIR, "prompts.json"), "w") as f:
        # Save just first 5 of each as examples
        examples = {fmt: prompts[fmt][:5] for fmt in prompts}
        json.dump(examples, f, indent=2, ensure_ascii=False)

    # Extract representations for each format
    for fmt in ["digits", "english", "french", "belgian"]:
        print(f"\n{'='*60}")
        print(f"Extracting representations for format: {fmt}")
        print(f"Example: {prompts[fmt][42]}")
        print(f"{'='*60}")

        hidden_states = extract_last_token_hidden_states(
            model, tokenizer, prompts[fmt], batch_size=64, device="cuda:0"
        )

        # Save per-layer representations
        fmt_dir = os.path.join(OUTPUT_DIR, fmt)
        os.makedirs(fmt_dir, exist_ok=True)

        for layer_idx, hidden in hidden_states.items():
            np.save(os.path.join(fmt_dir, f"layer_{layer_idx:02d}.npy"), hidden)

        print(f"Saved {len(hidden_states)} layers for {fmt}, shape per layer: {hidden_states[0].shape}")

        del hidden_states
        torch.cuda.empty_cache()

    # Save metadata
    metadata = {
        "model": MODEL_NAME,
        "num_numbers": len(data),
        "num_layers": model.config.num_hidden_layers + 1,
        "hidden_dim": model.config.hidden_size,
        "formats": list(prompts.keys()),
        "seed": SEED,
        "prompt_templates": {
            "digits": "The number is {n}.",
            "english": "The number is {english(n)}.",
            "french": "Le nombre est {french(n)}.",
            "belgian": "Le nombre est {belgian(n)}.",
        },
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone! All representations saved.")


if __name__ == "__main__":
    main()
