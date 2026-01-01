# Configuration settings

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 100,
        "lr": 1e-4,
        "seq_len": 192,
        "d_model": 512,
        "early_stop_patience": 30,
        "early_stop_min_delta": 0.001,
        "datasource": "Helsinki-NLP/opus_books",
        "lang_src": "el",
        "lang_tgt": "en",
        "model_folder": "results/transformer_weights",
        "model_basename": "tmodel_",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "results/runs/tmodel",
        "pretrained_model": "facebook/nllb-200-distilled-600M",
        "openai_model": "gpt-4o-mini",
        "openai_max_output_tokens": 256,
    }
