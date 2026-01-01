# Custom Transformer model
import torch
import torch.nn as nn
import math
from typing import Callable

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int,  vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the Transformer paper
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a positional encoding matrix of shape (max_seq_len, d_model) to hold the positional encodings
        # (Remember we do not care about the vocab dimension here)
        pe = torch.zeros(max_seq_len, d_model)

        # Create a tensor of shape (max_seq_len, 1) with values [0, 1, 2, ..., max_seq_len-1]
        position = torch.arange(0, max_seq_len).unsqueeze(1)

        # Create the div_term tensor for the denominator in the PE formula of shape (d_model/2,)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Apply the sine function to even indices in the array: sin(position / (10000^(2i/d_model)))
        pe[:, 0::2] = torch.sin(position * div_term) # shape: (max_seq_len, d_model/2) via broadcasting

        # Apply the cosine function to odd indices in the array: cos(position / (10000^(2i/d_model)))
        pe[:, 1::2] = torch.cos(position * div_term) # shape: (max_seq_len, d_model/2)

        # Add a batch dimension by unsqueezing at dim=0: shape (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register pe as a buffer to avoid being considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # Add positional encoding to input embeddings
        x = x + self.pe[:, :x.shape[1], :]  # pe shape: (batch_size, max_seq_len, d_model)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(d_model)) # bias is a learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True) # Mean over the last dimension, shape: (batch_size, seq_len, 1)
        std = x.std(dim=-1, keepdim=True) # Std over the last dimension, shape: (batch_size, seq_len, 1)

        # Normalize (eps added for numerical stability to avoid division by zero)
        normalized_x = (x - mean) / (std + self.eps) # shape: (batch_size, seq_len, d_model)

        # Scale and shift
        output = self.alpha * normalized_x + self.bias # shape: (batch_size, seq_len, d_model)

        return output

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class ResidualConnection(nn.Module):

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model), sublayer: (batch_size, seq_len, d_model)
        return x + self.dropout(sublayer(self.layer_norm(x)))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, dropout: nn.Dropout = None) -> torch.Tensor:
        d_k = query.shape[-1] # Get the dimension of d_k
        # Apply scaled dot-product attention according to the formula in the Transformer paper
        # (batch_size, num_heads, seq_len, d_k) @ (batch_size, num_heads, d_k, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None: # case when a mask is provided and needs to be applied
            # Apply the mask by setting masked positions to a very large negative value (-inf) before softmax so they become zero after softmax (this is for decoder self-attention)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf')) # shape: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1) # Apply softmax to get attention weights along the last dimension (seq_len), shape: (batch_size, num_heads, seq_len, seq_len)

        if dropout is not None: # case when dropout is provided
            attention_scores = dropout(attention_scores) # shape: (batch_size, num_heads, seq_len, seq_len)

        # Also return attention scores which can be used for visualization
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, d_k) -> (batch_size, num_heads, seq_len, d_k)
        return attention_scores @ value, attention_scores # shape: (batch_size, num_heads, seq_len, d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Calculate queries, keys and values for all heads in batch
        query = self.w_q(q) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        key = self.w_k(k)   # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

        # Split embeddings into num_heads
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # Calculate attention
        # We will also store the attention scores for visualization later, that is why we have self.attention_scores (save as attribute)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask=mask, dropout=self.dropout) # shape: (batch_size, num_heads, seq_len, d_k)

        # Concatenate heads
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        x =  x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k) # (x.shape[1] is seq_len after transpose)

        # Multiply by W_out, (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        x = self.w_out(x)

        return x

class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # Self-attention sublayer with residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # self.self_attention_block(x, x, x, src_mask) = self.self_attention_block.forward(x, x, x, mask), shape: (batch_size, seq_len, d_model)

        # Feed-forward sublayer with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x

class Encoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization(d_model)

    def forward (self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.layer_norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) # Self-attention with residual connection
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) # Cross-attention with residual connection
        x = self.residual_connections[2](x, self.feed_forward_block) # Feed-forward with residual connection
        return x

class Decoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.layer_norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        # Project the decoder output to the vocabulary size for generating probabilities over the vocabulary
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return self.linear(x)

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # Define some helper methods separately for inference purposes
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        enc_input = self.src_pos(self.src_embed(src))
        return self.encoder(enc_input, src_mask) # (batch_size, src_seq_len, d_model)

    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        dec_input = self.tgt_pos(self.tgt_embed(tgt))
        return self.decoder(dec_input, encoder_output, src_mask, tgt_mask) # (batch_size, tgt_seq_len, d_model)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(x) # (batch_size, tgt_seq_len, vocab_size)

    # Method for building the Transformer model
    @staticmethod
    def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6, dropout: float = 0.1) -> 'Transformer':

        # Create the embedding layers
        src_embed = InputEmbedding(d_model, src_vocab_size)
        tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

        # Create the positional encoding layers
        src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

        # Create the encoder blocks
        encoder_blocks = []
        for _ in range(num_layers):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)

        # Create the decoder blocks
        decoder_blocks = []
        for _ in range(num_layers):
            decoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
            decoder_blocks.append(decoder_block)

        # Create the encoder and decoder
        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

        # Create the projection layer
        projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        # Create the Transformer model
        transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

        # Initialize the weights of the model
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer

# Simple helper function to build a full Transformer directly
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6, dropout: float = 0.1) -> Transformer:
    return Transformer.build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model, num_heads, d_ff, num_layers, dropout)
