# Translates a sentence or dataset index using a pretrained Transformer
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import get_config

import warnings
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*use_auth_token.*", category=FutureWarning)

def load_pretrained_model(device, config):
    model_name = config["pretrained_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = "ell_Grek"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    return model, tokenizer

@torch.no_grad()
def translate_text(model, tokenizer, text, device, max_new_tokens=128):
    """Encode Greek text → tokens, generate English tokens, decode tokens → English text."""
    encoded = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    output_ids = model.generate(**encoded, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python src/translate_pretrained.py "Greek sentence"')
        print("  python src/translate_pretrained.py <dataset_index>")
        sys.exit(1)

    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_pretrained_model(device, config)

    user_input = sys.argv[1]

    # Case 1: dataset index
    if user_input.isdigit():
        idx = int(user_input)

        dataset = load_dataset(config["datasource"], f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")

        src_text = dataset[idx]["translation"][config["lang_src"]]
        tgt_text = dataset[idx]["translation"][config["lang_tgt"]]

        pred = translate_text(model, tokenizer, src_text, device)

        print("-" * 40)
        print(f"INDEX: {idx}")
        print(f"SOURCE: {src_text}")
        print(f"TARGET: {tgt_text}")
        print(f"PREDICTED: {pred}")

    # Case 2: raw text
    else:
        pred = translate_text(model, tokenizer, user_input, device)
        print(pred)


if __name__ == "__main__":
    main()
