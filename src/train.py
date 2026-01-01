# Train functions
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import torchmetrics
# from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text import SacreBLEUScore
from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
import os, time, random
from datetime import datetime

from model import build_transformer
from dataset import BilingualDataset
from config import get_config
from utils import get_all_sentences, get_model, causal_mask, save_splits, logger, append_jsonl

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_or_build_tokenizer(config, ds, lang):
    os.makedirs("dataset", exist_ok=True)
    tokenizer_path = os.path.join("dataset", f"tokenizer_{lang}.json")

    if not os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)

    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(config["datasource"], f"{config['lang_src']}-{config['lang_tgt']}", split="train", download_mode="reuse_dataset_if_exists")

    # Build tokenizers
    src_tokenizer = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tgt_tokenizer = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # print(f"Number of training samples: {len(train_ds_raw)}")
    # print(f"Number of validation samples: {len(train_ds_raw)}")

    # Save splits
    train_idx = train_ds_raw.indices
    val_idx = val_ds_raw.indices

    out_dir = os.path.join("..", "dataset") if os.getcwd().endswith("src") else "dataset"
    save_splits(ds_raw, train_idx, val_idx, config["lang_src"], config["lang_tgt"], out_dir)

    # Create the datasets
    train_ds = BilingualDataset(train_ds_raw, src_tokenizer, tgt_tokenizer, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, src_tokenizer, tgt_tokenizer, config["lang_src"], config["lang_tgt"], config["seq_len"])

    # Find the maximum length of each sentence in the src and tgt items
    max_len_src, max_len_tgt = 0, 0
    for item in ds_raw:
        src_ids = src_tokenizer.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tgt_tokenizer.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    # print(f"Max length src: {max_len_src}, tgt: {max_len_tgt}")

    # Create dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer

def greedy_decode(model, source, source_mask, tgt_tokenizer, max_len, device):
    # Goal: generate a translation token-by-token using greedy decoding (always pick the highest-logit next token)

    # Get [SOS] and [EOS] ids from the target tokenizer
    sos_idx = tgt_tokenizer.token_to_id("[SOS]")
    eos_idx = tgt_tokenizer.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

    while decoder_input.size(1) < max_len:

        # Build a casual mask for target, so decoder can’t look ahead
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Run the decoder: calculate next output/token (args: tgt, encoder_output, src_mask, tgt_mask)
        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        # Project last token to vocab logits
        logits = model.project(out[:, -1]) # out[:, -1] is (1, d_model) → logits (1, vocab)
        next_word = torch.argmax(logits, dim=1) # token id of shape: (1,)

        # Append the predicted token id to the decoder input (keep shape: 1 x current_len)
        next_token = next_word.view(1, 1) # (1, 1)
        decoder_input = torch.cat([decoder_input, next_token], dim=1)

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0) # shape: (seq_len,), e.g., tensor([SOS, 45, 932, 18, EOS, PAD, PAD, ...])

def run_validation(model, validation_ds, tgt_tokenizer, max_len, device, print_msg, global_step, writer, num_examples=2):
    # Goal: run the model on a few validation samples (default 2), print source/target/prediction, and log text metrics to TensorBoard
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # Shuffle validation batches to show different qualitative examples each epoch.
    # This affects only printed samples, not validation metrics.
    # val_batches = list(validation_ds)
    # random.shuffle(val_batches)

    with torch.no_grad():
        for batch in validation_ds:
            count += 1

            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch_size, 1, 1, seq_len)

            # Check that the batch_size is 1 (greedy_decode is written for a single sentence)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Generate translation using greedy decoding (autoregressive, one token at a time)
            model_out = greedy_decode(model, encoder_input, encoder_mask, tgt_tokenizer, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            # Cut the generated token sequence at the first [EOS] token
            # (prevents decoding padding or extra tokens beyond the end of the sentence)
            eos_id = int(tgt_tokenizer.token_to_id("[EOS]"))

            # Convert model output tensor -> Python list
            out_ids = model_out.detach().cpu().tolist()

            # If model output still has a batch dimension (shape: [1, T]), remove it
            out_ids = out_ids[0] if isinstance(out_ids[0], list) else out_ids

            # Ensure all token IDs are plain Python ints for safe EOS comparison
            out_ids = [int(x) for x in out_ids]

            # Stop decoding at EOS
            if eos_id in out_ids:
                out_ids = out_ids[:out_ids.index(eos_id) + 1]

            # Convert token ids to human-readable text
            model_out_text = tgt_tokenizer.decode(out_ids)

            # Always collect predictions for metrics
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # ONLY print first num_examples (print the source, target and model output)
            if count <= num_examples:
                print_msg('-' * 40)
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

    print_msg('-' * 40)

    # Compute validation metrics
    cer_metric = torchmetrics.CharErrorRate() # char error rate
    wer_metric = torchmetrics.WordErrorRate() # word error rate
    # bleu_metric = BLEUScore(n_gram=4, smooth=True) # BLEU metric
    bleu_metric = SacreBLEUScore(smooth=True, tokenize="13a")

    cer = cer_metric(predicted, expected)
    wer = wer_metric(predicted, expected)
    bleu = bleu_metric(predicted, [[e] for e in expected])
    # CER / WER ↓ better, BLEU ↑ better

    # TensorBoard logging
    # - X-axis (Step): global_step at end of epoch
    # - Y-axis: sequence-level evaluation metrics
    if writer:
        writer.add_scalar('validation cer', cer, global_step)
        writer.add_scalar('validation wer', wer, global_step)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

    # Save metrics to JSONL (one record per validation run / epoch)
    out_dir = os.path.join("..", "results") if os.getcwd().endswith("src") else "results"
    metrics_path = os.path.join(out_dir, "custom_transformer_metrics.jsonl")

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "custom_transformer",
        "global_step": int(global_step),
        "num_examples_printed": int(num_examples),
        "cer": float(cer),
        "wer": float(wer),
        "bleu": float(bleu),
        }
    append_jsonl(metrics_path, record)

    return bleu.item()

def train_model(config):
    # Log everything printed to console into results/train_eval_log.txt
    os.makedirs("results", exist_ok=True)
    logger("results/train_eval_log.txt")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if device == "cuda":
        cuda_idx = torch.cuda.current_device()
        print(f"Device name: {torch.cuda.get_device_name(cuda_idx)}")
        print(f"Device memory: {torch.cuda.get_device_properties(cuda_idx).total_memory / 1024 ** 3} GB")

    # Make sure the weights folder exists
    os.makedirs(config["model_folder"], exist_ok=True)

    # Data and model
    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_ds(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Tensorboard
    global_step = 0 # global_step counts optimizer updates (1 step = 1 training batch), total_steps ≈ num_epochs * [num_train_samples(train_ds_size) / batch_size]
    os.makedirs(os.path.dirname(config["experiment_name"]), exist_ok=True)
    writer = SummaryWriter(config['experiment_name'])

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    # Early stopping config
    patience = config["early_stop_patience"]
    min_delta = config["early_stop_min_delta"]

    best_state_dict = None
    best_epoch = None
    best_bleu = -1.0
    bad_epochs = 0

    # Timing and training
    total_training_time_sec = 0.0

    for epoch in range (config["num_epochs"]):
        epoch_start_time = time.time()

        if device == "cuda":
            torch.cuda.empty_cache()

        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch+1}")

        # The DataLoader splits the dataset into X mini-batches. Each mini-batch contains B samples (batch_size). Since the Dataset returns a dict, the DataLoader stacks each field across B samples:
        #   - encoder_input, decoder_input, label -> shape (B, seq_len)
        #   - encoder_mask, decoder_mask          -> shape (B, ...)
        # One (mini) batch here corresponds to one forward/backward pass and one optimizer step

        for batch in batch_iterator:

            # Get the input tensors and the label
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)
            label = batch['label'].to(device) # (B, seq_len)

            # DEBUG: print shift alignment ONCE (first batch of first epoch)
            # if epoch == 0 and global_step == 0:
            #     print("DECODER INPUT (first 10 ids):", decoder_input[0, :10].tolist())
            #     print("LABEL        (first 10 ids):", label[0, :10].tolist())

            # Forward pass: Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, vocab_size)

            # Compute the loss
            loss = loss_fn(
                proj_output.view(-1, proj_output.size(-1)), # (batch_size * seq_len, vocab_size)
                label.view(-1) # (batch_size * seq_len, )
            )

            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")

            # TensorBoard logging:
            # - X-axis (Step): global_step → one step = one training batch
            # - Y-axis (train loss): cross-entropy loss over target tokens (lower is better, absolute value depends on vocab size and label smoothing)
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backprop and weights update
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1

        # Run the validation at the end of each epoch
        val_bleu = run_validation(model, val_dataloader, tgt_tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Epoch and total timings
        epoch_time_sec = time.time() - epoch_start_time
        total_training_time_sec += epoch_time_sec

        # Early stopping
        if val_bleu > best_bleu + min_delta:
            best_bleu = val_bleu
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping. Best BLEU={best_bleu:.4f} at epoch {best_epoch}")
                break

        # Save the model at the end of every epoch
        model_filename = os.path.join(config["model_folder"], f"{config['model_basename']}{epoch:02d}.pt")
        torch.save({
            "epoch": epoch,
            "model_weights": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,

            # extra metadata
            "total_params": total_params,
            "trainable_params": trainable_params,
            "epoch_time_sec": epoch_time_sec,
            "total_training_time_sec": total_training_time_sec,
        }, model_filename)

    best_model_filename = os.path.join(
        config["model_folder"],
        f"{config['model_basename']}_best.pt"
    )

    torch.save({
        "best_epoch": best_epoch,
        "best_bleu": best_bleu,
        "model_weights": best_state_dict,
    }, best_model_filename)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
