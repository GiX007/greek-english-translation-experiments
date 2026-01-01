# Translation function
# Usage: python src/translate.py "Καλημέρα" or python src/translate.py 42
import os, time, sys
import torch

from config import get_config
from utils import get_model
from train import greedy_decode
from dataset import BilingualDataset

from datasets import load_dataset
from tokenizers import Tokenizer

def translate(sentence: str, max_len=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device)

    config = get_config()
    if max_len is None:
        max_len = config["seq_len"]

    t0 = time.time() # End-to-end inference timing starts here (tokenization → encoding → decoding → detokenization)

    # Load tokenizers
    tokenizer_src = Tokenizer.from_file(os.path.join("dataset", f"tokenizer_{config['lang_src']}.json"))
    tokenizer_tgt = Tokenizer.from_file(os.path.join("dataset", f"tokenizer_{config['lang_tgt']}.json"))

    # (Optional): numeric input → dataset index
    label = ""
    if isinstance(sentence, int) or (isinstance(sentence, str) and sentence.isdigit()):
        idx = int(sentence)

        ds_raw = load_dataset(config["datasource"], f"{config['lang_src']}-{config['lang_tgt']}", split="train")

        ds = BilingualDataset(ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

        sentence = ds[idx]["src_text"]
        label = ds[idx]["tgt_text"]

    # Build model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Load best weights
    weights_path = os.path.join("results", "transformer_weights", "tmodel__best.pt")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_weights"])

    # Translate the sentence
    model.eval()

    # Tokenize and encode source sentence
    src_ids = tokenizer_src.encode(sentence).ids
    src_ids = (
        [tokenizer_src.token_to_id("[SOS]")]
        + src_ids
        + [tokenizer_src.token_to_id("[EOS]")]
    )
    src_ids = src_ids[:max_len]
    src_ids += [tokenizer_src.token_to_id("[PAD]")] * (max_len - len(src_ids))

    encoder_input = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    encoder_mask = (encoder_input != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1)

    # Decode (greedy)
    t_decode = time.time() # decoding-only time (for greedy/beam search comparison)
    with torch.no_grad():
        output_ids = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

    decode_time = time.time() - t_decode

    # Detokenize
    output_ids = output_ids.tolist()
    if isinstance(output_ids[0], list):
        output_ids = output_ids[0]

    if tokenizer_tgt.token_to_id("[EOS]") in output_ids:
        output_ids = output_ids[: output_ids.index(tokenizer_tgt.token_to_id("[EOS]")) + 1]

    translation = tokenizer_tgt.decode(output_ids)

    total_time = time.time() - t0

    # Output
    if label:
        print("ID:       ", idx)
    if label:
        print("TARGET:   ", label)
    print("SOURCE:   ", sentence)
    print("PREDICTED:", translation)
    print("RAW OUTPUT IDS:", output_ids)
    print(f"Decoding time:     {decode_time:.3f} sec")
    print(f"End-to-end time:   {total_time:.3f} sec")

    return translation

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide a sentence to translate.")
    translate(sys.argv[1])
