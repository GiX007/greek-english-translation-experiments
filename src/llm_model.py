# OpenAI LLM model
# Uses the same validation split as the custom transformer.
#
# Note: Unlike the custom and pretrained transformer setups, we do not tokenize to ids ourselves here.
# We send raw text, and the OpenAI model uses its own internal tokenizer/vocabulary.
# See: https://platform.openai.com/docs/guides/text and https://platform.openai.com/docs/api-reference/responses.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

import torchmetrics
from torchmetrics.text import SacreBLEUScore

from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter

from openai import OpenAI

from utils import append_jsonl, logger
from config import get_config

from dotenv import load_dotenv
load_dotenv("OPENAI_API_KEY.env")

def build_validation_dataloader(config, val_indices):
    dataset = load_dataset(config["datasource"], f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")
    indices = val_indices.tolist() if hasattr(val_indices, "tolist") else list(val_indices)
    val_dataset = Subset(dataset, indices)
    return DataLoader(val_dataset, batch_size=1, shuffle=False)

def _openai_translate_one(client: OpenAI, model_name: str, source_text: str, max_output_tokens: int = 256):
    """
    Translation via OpenAI Responses API.
    We enforce a clean comparison by using:
      - temperature=0 (more deterministic)
      - output only the translation text (no explanations)
    """
    system_msg = (
        "You are a translation engine. Translate Greek to English.\n"
        "Return ONLY the English translation. No extra words, no quotes, no punctuation added."
    )

    # Responses API: https://platform.openai.com/docs/api-reference/responses
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": source_text},
        ],
        temperature=0,
        max_output_tokens=max_output_tokens,
    )

    return (resp.output_text or "").strip()

def openai_translate_with_retries(client: OpenAI, model_name: str, source_text: str, max_output_tokens: int = 256, max_retries: int = 6, base_sleep: float = 1.0) -> str:
    """
    Why retries?
    - Custom + HF models run locally, so they don't face network/rate-limit failures.
    - OpenAI calls can fail transiently (429 rate limit, timeouts, occasional 5xx).
    - Retrying does NOT change model behavior/decoding: we keep temperature=0 and the same prompt.
      It only ensures we actually get a response instead of the whole evaluation crashing mid-run.
    """
    for attempt in range(max_retries):
        try:
            return _openai_translate_one(client=client, model_name=model_name, source_text=source_text, max_output_tokens=max_output_tokens)
        except Exception:
            if attempt == max_retries - 1:
                raise time.sleep(base_sleep * (2 ** attempt))

@torch.no_grad()
def run_validation_openai(validation_loader, config, print_msg=print, global_step=0, writer=None):
    """
    Equivalent validation loop:
      - iterate validation samples (batch_size = 1)
      - translate each source sentence using OpenAI
      - print SOURCE / TARGET / PREDICTED for first few examples
      - accumulate predictions
      - compute CER / WER / SacreBLEU
      - append metrics to JSONL
    """
    # OpenAI client reads API key from env var OPENAI_API_KEY by default
    client = OpenAI()

    count = 0
    predicted = []
    expected = []

    for batch in validation_loader:
        count += 1

        source_text = batch["translation"][config["lang_src"]][0]
        target_text = batch["translation"][config["lang_tgt"]][0]

        pred_text = openai_translate_with_retries(client=client, model_name=config["openai_model"], source_text=source_text, max_output_tokens=config.get("openai_max_output_tokens", 256))

        predicted.append(pred_text)
        expected.append(target_text)

        if count <= 2:
            print_msg("-" * 40)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{pred_text}")

    print_msg("-" * 40)

    # Metrics (same style as your HF eval)
    cer_metric = torchmetrics.text.CharErrorRate()
    wer_metric = torchmetrics.text.WordErrorRate()
    bleu_metric = SacreBLEUScore(smooth=True, tokenize="13a")

    cer = cer_metric(predicted, expected)
    wer = wer_metric(predicted, expected)
    bleu = bleu_metric(predicted, [[e] for e in expected])

    if writer:
        writer.add_scalar("openai/validation cer", cer, global_step)
        writer.add_scalar("openai/validation wer", wer, global_step)
        writer.add_scalar("openai/validation BLEU", bleu, global_step)
        writer.flush()

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "openai_llm_metrics.jsonl")

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "openai_llm",
        "openai_model": config["openai_model"],
        "decode": "temperature_0",
        "global_step": int(global_step),
        "cer": float(cer),
        "wer": float(wer),
        "bleu": float(bleu),
        "n_samples": len(expected),
    }
    append_jsonl(metrics_path, record)

    return {"bleu": float(bleu), "wer": float(wer), "cer": float(cer)}


def main():
    config = get_config()

    os.makedirs("results", exist_ok=True)
    logger("results/openai_llm_eval_log.txt")

    os.makedirs(os.path.dirname(config["experiment_name"]), exist_ok=True)
    writer = SummaryWriter(config["experiment_name"] + "_openai")

    print("Using OpenAI model:", config["openai_model"])

    # Load validation indices (same split)
    val_indices_path = r"C:\Users\giorg\Projects\PycharmProjects\greek-english-translation-experiments\dataset\val_indices.pt"
    val_indices = torch.load(val_indices_path, weights_only=True)

    # Build validation DataLoader using explicit indices
    val_loader = build_validation_dataloader(config, val_indices)

    global_step = int(datetime.now().timestamp())
    metrics = run_validation_openai(
        validation_loader=val_loader,
        config=config,
        print_msg=print,
        global_step=global_step,
        writer=writer
    )
    writer.close()

    print("-" * 40)
    print("OpenAI LLM — Validation Metrics")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"WER : {metrics['wer']:.4f}")
    print(f"CER : {metrics['cer']:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()
