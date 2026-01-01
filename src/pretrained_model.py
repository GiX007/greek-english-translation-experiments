# Pretrained model evaluation (EL → EN).
# Uses the same validation split as the custom transformer.
#
# Pretrained model used:
#   facebook/nllb-200-distilled-600M
#   - Source language: Greek (ell_Grek)
#   - Target language: English (eng_Latn)
import os
import warnings
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import torchmetrics
from torchmetrics.text import SacreBLEUScore

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils import append_jsonl, logger
from config import get_config


# Silence non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def build_validation_dataloader(config, val_indices):
    dataset = load_dataset(config["datasource"], f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")

    indices = val_indices.tolist() if hasattr(val_indices, "tolist") else list(val_indices)
    val_dataset = Subset(dataset, indices)

    return DataLoader(val_dataset, batch_size=1, shuffle=False)

@torch.no_grad()
def generate_translation(model, tokenizer, source_text, device, max_new_tokens):
    """
    1. Encode source Greek text → token IDs
    2. Generate target English token IDs using greedy decoding
    3. Return generated token IDs
    """
    encoded = tokenizer(source_text, return_tensors="pt", truncation=True).to(device)
    output_ids = model.generate(**encoded, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])

    return output_ids[0]

@torch.no_grad()
def run_validation_hf(model, tokenizer, validation_loader, config, device, print_msg=print, global_step=0, writer=None, num_examples=2):
    """
    Run a full validation pass:
      - encode Greek text
      - generate English translation
      - decode tokens → text
      - compute BLEU / WER / CER
    """
    predicted = []
    expected = []
    count = 0

    for batch in validation_loader:
        count += 1

        source_text = batch["translation"][config["lang_src"]][0]
        target_text = batch["translation"][config["lang_tgt"]][0]

        out_ids = generate_translation(
            model,
            tokenizer,
            source_text,
            device,
            max_new_tokens=config["seq_len"],
        )

        pred_text = tokenizer.decode(out_ids, skip_special_tokens=True)

        predicted.append(pred_text)
        expected.append(target_text)

        if count <= num_examples:
            print_msg("-" * 40)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{pred_text}")

    print_msg("-" * 40)

    # Metrics
    cer_metric = torchmetrics.text.CharErrorRate()
    wer_metric = torchmetrics.text.WordErrorRate()
    bleu_metric = SacreBLEUScore(smooth=True, tokenize="13a")

    cer = cer_metric(predicted, expected)
    wer = wer_metric(predicted, expected)
    bleu = bleu_metric(predicted, [[e] for e in expected])

    # TensorBoard logging
    if writer:
        writer.add_scalar("pretrained/validation BLEU", bleu, global_step)
        writer.add_scalar("pretrained/validation WER", wer, global_step)
        writer.add_scalar("pretrained/validation CER", cer, global_step)
        writer.flush()

    # Save metrics to JSONL
    os.makedirs("results", exist_ok=True)
    metrics_path = os.path.join("results", "pretrained_metrics.jsonl")

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "pretrained",
        "pretrained_model": config["pretrained_model"],
        "decode": "greedy",
        "cer": float(cer),
        "wer": float(wer),
        "bleu": float(bleu),
    }
    append_jsonl(metrics_path, record)

    return {
        "bleu": float(bleu),
        "wer": float(wer),
        "cer": float(cer),
    }

def main():
    config = get_config()

    # Log everything printed to console
    os.makedirs("results", exist_ok=True)
    logger("results/pretrained_eval_log.txt")

    # TensorBoard
    os.makedirs(os.path.dirname(config["experiment_name"]), exist_ok=True)
    writer = SummaryWriter(config["experiment_name"] + "_hf")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load validation indices
    val_indices_path = r"C:\Users\giorg\Projects\PycharmProjects\greek-english-translation-experiments\dataset\val_indices.pt"
    try:
        val_indices = torch.load(val_indices_path, weights_only=True)
    except TypeError:
        val_indices = torch.load(val_indices_path)

    # Load NLLB pretrained model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", token=True)
    tokenizer.src_lang = "ell_Grek"

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", token=True).to(device)
    model.eval()

    # Sanity check
    test = "Το σχολείο μας έχει εννιά τάξεις."
    out = generate_translation(model, tokenizer, test, device, max_new_tokens=64)
    print("SOURCE:", test)
    print("SANITY CHECK:", tokenizer.decode(out, skip_special_tokens=True))

    # Validation
    val_loader = build_validation_dataloader(config, val_indices)

    global_step = int(datetime.now().timestamp())
    metrics = run_validation_hf(
        model=model,
        tokenizer=tokenizer,
        validation_loader=val_loader,
        config=config,
        device=device,
        print_msg=print,
        global_step=global_step,
        writer=writer,
    )

    writer.close()

    print("-" * 40)
    print("Pretrained Model — Validation Metrics")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"WER : {metrics['wer']:.4f}")
    print(f"CER : {metrics['cer']:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()
