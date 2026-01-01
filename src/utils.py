# Utility functions
import os, glob, json, sys
import torch
from model import build_transformer
from types import SimpleNamespace

def latest_weights_file_path(config):
    model_folder = config["model_folder"]
    model_filename=  f"{config['model_basename']}*"
    weights_files = glob.glob(os.path.join(model_folder, model_filename))

    if len(weights_files) == 0:
        return None

    weights_files.sort()
    return weights_files[-1]

def causal_mask(size):
    mask = torch.tril(torch.ones((1, size, size), dtype=torch.bool))
    return mask

def get_all_sentences(ds, lang):
    return [item["translation"][lang] for item in ds]

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def save_splits(ds_raw, train_idx, val_idx, src_lang, tgt_lang, out_dir="dataset"):
    os.makedirs(out_dir, exist_ok=True)

    # Save indices (for rebuilding splits and dataloaders later for any model)
    torch.save(train_idx, os.path.join(out_dir, "train_indices.pt"))
    torch.save(val_idx, os.path.join(out_dir, "val_indices.pt"))

    def write_jsonl(indices, path):
        with open(path, "w", encoding="utf-8") as f:
            for i in indices:
                item = ds_raw[int(i)]
                src = item["translation"][src_lang]
                tgt = item["translation"][tgt_lang]
                f.write(json.dumps({"src": src, "tgt": tgt}, ensure_ascii=False) + "\n")

    write_jsonl(train_idx, os.path.join(out_dir, "train.jsonl"))
    write_jsonl(val_idx, os.path.join(out_dir, "val.jsonl"))

def logger(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    original_stdout = sys.stdout
    f = open(filepath, "w", encoding="utf-8")

    def write(msg):
        original_stdout.write(msg)
        f.write(msg)

    def flush():
        original_stdout.flush()
        f.flush()

    # Create a minimal object with write/flush (no class)
    sys.stdout = SimpleNamespace(write=write, flush=flush)

def append_jsonl(path, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
