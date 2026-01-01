# Get the dataset class for loading and processing data
import torch
from torch.utils.data import Dataset
from utils import causal_mask

class BilingualDataset(Dataset):

    def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, max_seq_len):
        super().__init__()
        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_seq_len = max_seq_len

        # Define special tokens
        self.sos_token = torch.tensor([tgt_tokenizer.token_to_id("[SOS]")], dtype=torch.long) # Start of Sequence token
        self.eos_token = torch.tensor([tgt_tokenizer.token_to_id("[EOS]")], dtype=torch.long) # End of Sequence token
        self.pad_token = torch.tensor([tgt_tokenizer.token_to_id("[PAD]")], dtype=torch.long) # Padding token

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        src_text = item['translation'][self.src_lang]
        tgt_text = item['translation'][self.tgt_lang]

        # Tokenize source and target texts
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokens = self.tgt_tokenizer.encode(tgt_text).ids

        # Calculate padding lengths
        enc_padding_tokens = self.max_seq_len - len(enc_input_tokens) - 2 # for [SOS] and [EOS]
        dec_padding_tokens = self.max_seq_len - len(dec_input_tokens) - 1 # for [SOS], [EOS] is not known during input

        # Make sure padding tokens are not negative (in case of long sequences)
        if enc_padding_tokens < 0 or dec_padding_tokens < 0:
            raise ValueError("Sentence is too long.")

        # Add special tokens and padding
        enc_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.long),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_padding_tokens, dtype=torch.long),
        ],
        dim=0
        ) # Encoder input: [SOS] ...tokens... [EOS] ...padding..., shape: (max_seq_len,)

        dec_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.long),
            torch.tensor([self.pad_token] * dec_padding_tokens, dtype=torch.long),
        ],
        dim=0
        ) # Decoder input: [SOS] ...tokens... ...padding..., shape: (max_seq_len,)

        # Define the decoder target (label) sequence
        # Decoder target is the decoder input shifted left by one position with [EOS] at the end (paper). This happens because during training, the model learns to predict the next token
        # at each time step and the target sequence should be shifted by one position, so that the model learns to predict the token following the current input token.
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.long),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_padding_tokens, dtype=torch.long),
        ],
        dim=0
        ) # Decoder target: ...tokens... [EOS] ...padding..., shape: (max_seq_len,)

        # Double check shapes
        assert enc_input.shape[0] == self.max_seq_len, f"Encoder input length mismatch: {enc_input.shape[0]} vs {self.max_seq_len}"
        assert dec_input.shape[0] == self.max_seq_len, f"Decoder input length mismatch: {dec_input.shape[0]} vs {self.max_seq_len}"
        assert label.shape[0] == self.max_seq_len, f"Label length mismatch: {label.shape[0]} vs {self.max_seq_len}"


        return {
            "encoder_input": enc_input, # shape: (max_seq_len,)
            "decoder_input": dec_input, # shape: (max_seq_len,)

            # Mask padding tokens in encoder self-attention
            "encoder_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0), # shape: (1, 1, max_seq_len)

            # Mask future tokens and padding tokens in decoder self-attention
            "decoder_mask": (dec_input != self.pad_token).unsqueeze(0) & causal_mask(dec_input.shape[0]), # shapes: (1, max_seq_len) & (1, max_seq_len, max_seq_len)

            "label": label, # shape: (max_seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
