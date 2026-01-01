# Sanity checks
import os
import torch
import torch.nn as nn
from model import InputEmbedding, PositionalEncoding, LayerNormalization, FeedForwardBlock, ResidualConnection, MultiHeadAttentionBlock, EncoderBlock, Encoder, DecoderBlock, Decoder, ProjectionLayer, Transformer

def test_input_embedding():
    print("=== Test InputEmbeddings ===")
    d_model = 8
    vocab_size = 20

    # tiny batch: 2 sequences, each of length 4, token ids in [0, vocab_size)
    x = torch.tensor([
        [1, 2, 3, 14],
        [5, 6, 7, 18]
    ]) # shape: (2, 4)

    emb = InputEmbedding(d_model=d_model, vocab_size=vocab_size)
    out = emb(x) # shape: (2, 4, d_model)

    print("Input shape:", x.shape) # (2, 4)
    print("Input:\n", x)  # tensor with token ids
    print("First sample's tokens:", x[0])  # tensor with token ids of first sample of shape (4,)
    print("Output shape:", out.shape) # (2, 4, d_model)
    print("Output:\n", out)  # tensor with embeddings
    print("First sample's embeddings:\n", out[0])  # embeddings of first sample of shape (4, d_model)
    print()

def test_input_embeddings_with_vocab():
    print("=== Test InputEmbeddings (Human-Friendly) ===")
    d_model = 8
    vocab = {
        "hello": 0,
        "world": 1,
        "ai": 2,
        "rocks": 3
    }

    # tiny batch: 2 sequences, each length 2
    x = torch.tensor([
        [vocab["hello"], vocab["world"]],
        [vocab["ai"], vocab["rocks"]]
    ])  # shape: (2, 2)

    emb = InputEmbedding(d_model=d_model, vocab_size=len(vocab))
    out = emb(x)  # shape: (2, 2, d_model)

    print("Input:\n", x)  # tensor with token ids
    print("Input shape:", x.shape)  # (2, 2)
    print("Token of 'hello':", x[0,0].item())  # token id for "hello"
    print("Output shape:", out.shape)  # (2, 2, d_model)
    print("Output:\n", out)  # tensor with embeddings of shape (2, 2, d_model)
    print("Embedding for 'hello':\n", out[0,0])  # embedding for "hello"
    print()

def test_positional_encoding():
    print("=== Test PositionalEncoding ===")
    d_model = 8
    seq_len = 4
    dropout = 0.0  # set 0.0 to see exact values

    # fake embeddings: batch_size=2, max_seq_len=4, d_model=8
    x = torch.zeros(2, seq_len, d_model)

    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len, dropout=dropout)
    out = pe(x)

    print("Input shape:", x.shape) # (2, 4, 8)
    print("Output shape:", out.shape) # (2, 4, 8)
    print("Positional encoding for position 0:")
    print(out[0, 0]) # same as pe.pe[0, 0]
    print("Positional encoding for position 1:")
    print(out[0, 1])

    print("\n=== Broadcasting Explanation ===")
    position = torch.arange(0, seq_len).unsqueeze(1) # (4,1)
    div_term = torch.arange(0, d_model, 2).float() # (4,)

    print("position (shape:", position.shape, ")")
    print(position)
    print()

    print("div_term (shape:", div_term.shape, "):")
    print(div_term)
    print()

    multiplied = position * div_term # broadcasting here -> (4,4)

    print("position * div_term (shape:", multiplied.shape, ")")
    print(multiplied)
    print()

def test_embeddings_plus_positional():
    print("=== Test Embeddings + PositionalEncoding ===")
    d_model = 8
    vocab_size = 20
    seq_len = 4
    dropout = 0.0 # set 0.0 to see exact values

    # tiny batch of token ids
    x_tokens = torch.tensor([
        [1, 2, 3, 14],
        [5, 6, 7, 18]
    ]) # shape: (2, 4)

    emb = InputEmbedding(d_model=d_model, vocab_size=vocab_size)
    pe = PositionalEncoding(d_model=d_model, max_seq_len=seq_len, dropout=dropout)

    x = emb(x_tokens) # shape: (2, 4, d_model)
    out = pe(x) # shape: (2, 4, d_model)

    print("Input token ids:\n", x_tokens)
    print("Tokens shape:", x_tokens.shape) # (2, 4)
    print("Embeddings shape:", x.shape) # (2, 4, d_model)
    print("Output shape (embeddings + pe):", out.shape) # (2, 4, d_model)
    print("First sequence, first token embedding:\n", x[0,0])
    print("First sequence, first token embedding + pe:\n", out[0,0])
    print()

def test_layer_normalization():
    print("=== Test LayerNormalization ===")
    d_model = 8
    seq_len = 4
    batch_size = 2

    x = torch.randn(batch_size, seq_len, d_model) # random tensor of shape (2, 4, 8)

    ln = LayerNormalization(d_model=d_model)
    out = ln(x)

    print("Input shape:", x.shape) # (2, 4, 8)
    print("Output shape:", out.shape) # (2, 4, 8)

    # Check a single position normalization
    print("Input (first sample, first position):", x[0,0])
    print("Normalized Output (first sample, first position):", out[0,0])
    print("Mean of normalized output (should be ~0):", out[0,0].mean().item())
    print("Std of normalized output (should be ~1):", out[0,0].std().item())
    print()

def test_feedforward_block():
    print("=== Test FeedForwardBlock ===")
    d_model = 8
    d_ff = 16
    dropout = 0.1
    batch_size = 2
    seq_len = 4

    x = torch.randn(batch_size, seq_len, d_model) # random tensor of shape (2, 4, 8)
    ffb = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)

    print("Input shape:", x.shape) # (2, 4, 8)
    print("Input sample (first token vector):")
    print(x[0, 0])

    # ----- step-by-step forward -----
    x1 = ffb.linear_1(x)
    print("\nAfter Linear 1 shape:", x1.shape) # (2, 4, 16)
    print("After Linear 1 sample (first token vector):")
    print(x1[0, 0])

    x_relu = torch.relu(x1)
    print("\nAfter ReLU shape:", x_relu.shape) # (2, 4, 16)
    print("After ReLU sample (first token vector):")
    print(x_relu[0, 0])

    x_drop = ffb.dropout(x_relu)
    print("\nAfter Dropout shape:", x_drop.shape)  # (2, 4, 16)
    print("After Dropout sample (first token vector):")
    print(x_drop[0, 0])

    out = ffb.linear_2(x_drop)
    print("\nFinal Output shape:", out.shape)  # (2, 4, 8)
    print("Output sample (first token vector):")
    print(out[0, 0])
    print()

def test_residual_connection():
    print("=== Test ResidualConnection ===")
    d_model = 8
    dropout = 0.0 # set to 0.0 to see exact values
    batch_size = 2
    seq_len = 4

    x = torch.randn(batch_size, seq_len, d_model) # random tensor of shape (2, 4, 8)
    residual = ResidualConnection(d_model=d_model, dropout=dropout)

    print("Input x shape:", x.shape)
    print("Input sample (first token vector):")
    print(x[0, 0])
    print()

    # Define a dummy sublayer
    sublayer = nn.Linear(d_model, d_model)

    # Step 1: LayerNorm
    x_norm = residual.layer_norm(x)
    print("After LayerNorm shape:", x_norm.shape)
    print("Normalized sample (first token vector):")
    print(x_norm[0, 0])
    print()

    # Step 2: Sublayer
    sublayer_out = sublayer(x_norm)
    print("After Sublayer shape:", sublayer_out.shape)
    print("Sublayer output sample (first token vector):")
    print(sublayer_out[0, 0])
    print()

    # Step 3: Dropout
    sublayer_out_drop = residual.dropout(sublayer_out)
    print("After Dropout shape:", sublayer_out_drop.shape)
    print("After Dropout sample (first token vector):")
    print(sublayer_out_drop[0, 0])
    print()

    # Step 4: Residual Addition
    out = x + sublayer_out_drop
    print("Final Output shape:", out.shape)
    print("Output sample (first token vector after residual + sublayer):")
    print(out[0, 0])
    print()

    # Full forward pass
    out_full = residual(x, sublayer)
    print("Output from full forward pass shape:", out_full.shape)
    print("Output from full forward pass sample (first token vector):")
    print(out_full[0, 0])
    print("Forward pass output matches manual?", torch.allclose(out, out_full))
    print()

def test_multihead_attention_block():
    print("=== Test MultiHeadAttentionBlock ===")
    d_model = 8
    h = 2
    dropout = 0.0 # set to 0.0 to see exact values
    batch_size = 2
    seq_len = 4

    mha = MultiHeadAttentionBlock(d_model=d_model, num_heads=h, dropout=dropout)

    # Input tensor for self attention
    x = torch.randn(batch_size, seq_len, d_model) # random tensor of shape (2, 4, 8)

    print("Input x shape:", x.shape) # (2,4,8)
    print("Input sample (first token vector):")
    print(x[0, 0])
    print()

    # Linear projections
    q = mha.w_q(x)
    k = mha.w_k(x)
    v = mha.w_v(x)

    print("After linear projections shapes - Q:", q.shape, "K:", k.shape, "V:", v.shape) # (2,4,8)
    print("Q sample (first token vector):")
    print(q[0, 0])
    print()

    # Split into heads
    d_k = mha.d_k
    print(f"d_model={d_model}, heads={h}, d_k={d_k}")

    q_heads = q.view(batch_size, h, seq_len, d_k) # (batch_size, h, seq_len, d_k)
    k_heads = k.view(batch_size, h, seq_len, d_k) # (batch_size, h, seq_len, d_k)
    v_heads = v.view(batch_size, h, seq_len, d_k) # (batch_size, h, seq_len, d_k)

    print("After splitting into heads shapes - Q_heads:", q_heads.shape, "K_heads:", k_heads.shape, "V_heads:", v_heads.shape) # (2,h,4,d_k)
    print("Q_heads sample (first head, first token vector):")
    print(q_heads[0, 0, 0])
    print()

    # Attention
    attn_output, attn_scores = mha.attention(q_heads, k_heads, v_heads, mask=None, dropout=mha.dropout)

    print("=== Attention() outputs ===")
    print("attn_output shape:", attn_output.shape)
    print("attn_scores shape:", attn_scores.shape)
    print()

    print("attn_output meaning:")
    print("  → This is the weighted sum of values per head, per token.")
    print("  → Shape: (batch, heads, seq_len, d_k)")
    print("Sample attn_output for batch 0, head 0, token 0:")
    print(attn_output[0, 0, 0])
    print()

    print("attn_scores meaning:")
    print("  → This is the attention distribution (weights/softmax probabilities).")
    print("  → Each row sums to 1 across seq_len.")
    print("Sample attention scores for batch 0, head 0, query token 0:")
    print(attn_scores[0, 0, 0])
    print("Sum of these scores (should be 1):", attn_scores[0, 0, 0].sum().item())
    print()

    # Concatenate heads
    attn_output_concat = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model) # (2, 4, 8)
    print("After concatenating heads shape:", attn_output_concat.shape) # (2,4,8)
    print("Concatenated output sample (first token vector):")
    print(attn_output_concat[0, 0])
    print()

    # Final linear layer
    out = mha.w_out(attn_output_concat) # (2, 4, 8)
    print("Final output shape after w_out:", out.shape) # (2,4,8)
    print("Final output sample (first token vector):")
    print(out[0, 0])
    print()

    # Verify full forward pass
    out_full = mha(x, x, x, mask=None)
    print("Output from full forward pass shape:", out_full.shape) # (2,4,8)
    print("Output from full forward pass sample (first token vector):")
    print(out_full[0, 0])
    print("Forward pass output matches manual?", torch.allclose(out, out_full))
    print()

def test_encoder_block():
    print("=== Test EncoderBlock ===")
    d_model = 8
    d_ff = 16
    h = 2
    dropout = 0.0          # set 0.0 so manual and forward match
    batch_size = 2
    seq_len = 4

    # Components
    self_attn = MultiHeadAttentionBlock(d_model=d_model, num_heads=h, dropout=dropout)
    ffb = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
    encoder_block = EncoderBlock(d_model=d_model, self_attention_block=self_attn, feed_forward_block=ffb, dropout=dropout)

    # Input
    x = torch.randn(batch_size, seq_len, d_model) # random tensor of shape (2, 4, 8)
    src_mask = None  # No mask for this test

    print("Input x shape:", x.shape)
    print("Input sample (first token):")
    print(x[0, 0])
    print()

    # Manual Forward Pass (step by step)

    # Residual 0: LayerNorm -> Self-Attention -> Dropout -> +x
    res0 = encoder_block.residual_connections[0]

    x_norm0 = res0.layer_norm(x)
    print("After first LayerNorm shape:", x_norm0.shape)
    print("First LayerNorm sample (first token):")
    print(x_norm0[0, 0])
    print()

    sa_out = encoder_block.self_attention_block(x_norm0, x_norm0, x_norm0, src_mask)
    print("After Self-Attention shape:", sa_out.shape)
    print("Self-Attention output sample (first token):")
    print(sa_out[0, 0])
    print()

    sa_drop = res0.dropout(sa_out)
    x1 = x + sa_drop
    print("After first Residual Addition shape:", x1.shape)
    print("After first Residual Addition sample (first token):")
    print(x1[0, 0])
    print()

    # Residual 1: LayerNorm -> FeedForward -> Dropout -> +x1

    res1 = encoder_block.residual_connections[1]

    x_norm1 = res1.layer_norm(x1)
    print("After second LayerNorm shape:", x_norm1.shape)
    print("Second LayerNorm sample (first token):")
    print(x_norm1[0, 0])
    print()

    ffb_out = encoder_block.feed_forward_block(x_norm1)
    print("After FeedForward shape:", ffb_out.shape)
    print("FeedForward output sample (first token):")
    print(ffb_out[0, 0])
    print()

    ffb_drop = res1.dropout(ffb_out)
    out = x1 + ffb_drop
    print("After second Residual Addition shape:", out.shape)
    print("After second Residual Addition sample (first token):")
    print(out[0, 0])
    print()

    # Full forward pass
    out_full = encoder_block(x, src_mask)
    print("Output from full forward pass shape:", out_full.shape)
    print("Output from full forward pass sample (first token):")
    print(out_full[0, 0])
    print("Forward pass output matches manual?", torch.allclose(out, out_full))
    print()

def test_encoder():
    print("=== Test Encoder ===")
    d_model = 8
    d_ff = 16
    h = 2
    dropout = 0.0 # again, 0.0 for deterministic behavior
    batch_size = 2
    seq_len = 4
    num_layers = 2

    # Build list of EncoderBlocks
    layers = nn.ModuleList([
        EncoderBlock(
            d_model=d_model,
            self_attention_block=MultiHeadAttentionBlock(d_model=d_model, num_heads=h, dropout=dropout),
            feed_forward_block=FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout),
            dropout=dropout
        )
        for _ in range(num_layers)
    ])

    encoder = Encoder(d_model=d_model, layers=layers)

    # Input
    x = torch.randn(batch_size, seq_len, d_model) # random tensor of shape (2, 4, 8)
    src_mask = None  # No mask for this test

    print("Input x shape:", x.shape)
    print("Input sample (first token):")
    print(x[0, 0])
    print()

    # Manual forward pass through all layers
    x_manual = x
    for i, layer in enumerate(encoder.layers):
        print(f"--- Encoder Layer {i+1} ---")
        x_manual = layer(x_manual, src_mask)
        print("Output shape after this layer:", x_manual.shape)
        print("Output sample (first token):")
        print(x_manual[0, 0])
        print()

    # Final LayerNorm
    x_manual_norm = encoder.layer_norm(x_manual)
    print("After final LayerNorm shape:", x_manual_norm.shape)
    print("After final LayerNorm sample (first token):")
    print(x_manual_norm[0, 0])
    print()

    # Encoder forward pass
    encoder_out = encoder(x, src_mask)
    print("Output from Encoder forward pass shape:", encoder_out.shape)
    print("Output from Encoder forward pass sample (first token):")
    print(encoder_out[0, 0])
    print()

    # Compare manual and forward pass
    print("Forward pass output matches manual?", torch.allclose(x_manual_norm, encoder_out))
    print()

def test_decoder_block():
    print("=== Test DecoderBlock ===")
    d_model = 8
    d_ff = 16
    h = 2
    dropout = 0.0 # set 0.0 so manual and forward match
    batch_size = 2
    src_seq_len = 5
    tgt_seq_len = 4

    # Components
    self_attn = MultiHeadAttentionBlock(d_model=d_model, num_heads=h, dropout=dropout)
    cross_attn = MultiHeadAttentionBlock(d_model=d_model, num_heads=h, dropout=dropout)
    ffb = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
    decoder_block = DecoderBlock(d_model=d_model, self_attention_block=self_attn, cross_attention_block=cross_attn, feed_forward_block=ffb, dropout=dropout)

    # Input
    x = torch.randn(batch_size, tgt_seq_len, d_model) # random tensor of shape (2, 4, 8)
    enc_output = torch.randn(batch_size, src_seq_len, d_model) # random tensor of shape (2, 5, 8)
    tgt_mask = None  # No mask for this test
    src_mask = None  # No mask for this test

    print("Decoder Input x shape:", x.shape)
    print("Encoder Output shape:", enc_output.shape)
    print("Decoder Input sample (first token):")
    print(x[0, 0])
    print()

    # Residual 0: LayerNorm -> Self-Attention -> Dropout -> +x
    res0 = decoder_block.residual_connections[0]

    x_norm0 = res0.layer_norm(x)
    print("After first LayerNorm shape:", x_norm0.shape)
    print("First LayerNorm sample (first token):")
    print(x_norm0[0, 0])
    print()

    sa_out = decoder_block.self_attention_block(x_norm0, x_norm0, x_norm0, tgt_mask)
    print("After Self-Attention shape:", sa_out.shape)
    print("Self-Attention output sample (first token):")
    print(sa_out[0, 0])
    print()

    sa_drop = res0.dropout(sa_out)
    x1 = x + sa_drop
    print("After first Residual Addition shape:", x1.shape)
    print("After first Residual Addition sample (first token):")
    print(x1[0, 0])
    print()

    # Residual 1: LayerNorm -> Cross-Attention -> Dropout -> +x1
    res1 = decoder_block.residual_connections[1]

    x_norm1 = res1.layer_norm(x1)
    print("After second LayerNorm shape:", x_norm1.shape)
    print("Second LayerNorm sample (first token):")
    print(x_norm1[0, 0])
    print()

    ca_out = decoder_block.cross_attention_block(x_norm1, enc_output, enc_output, src_mask)
    print("After Cross-Attention shape:", ca_out.shape)
    print("Cross-Attention output sample (first token):")
    print(ca_out[0, 0])
    print()

    ca_drop = res1.dropout(ca_out)
    x2 = x1 + ca_drop
    print("After second Residual Addition shape:", x2.shape)
    print("After second Residual Addition sample (first token):")
    print(x2[0, 0])
    print()

    # Residual 2: LayerNorm -> FeedForward -> Dropout -> +x2
    res2 = decoder_block.residual_connections[2]

    x_norm2 = res2.layer_norm(x2)
    print("After third LayerNorm shape:", x_norm2.shape)
    print("Third LayerNorm sample (first token):")
    print(x_norm2[0, 0])
    print()

    ffb_out = decoder_block.feed_forward_block(x_norm2)
    print("After FeedForward shape:", ffb_out.shape)
    print("FeedForward output sample (first token):")
    print(ffb_out[0, 0])
    print()

    ffb_drop = res2.dropout(ffb_out)
    out = x2 + ffb_drop
    print("After third Residual Addition shape:", out.shape)
    print("After third Residual Addition sample (first token):")
    print(out[0, 0])
    print()

    # Full forward pass
    out_full = decoder_block(x, enc_output, src_mask, tgt_mask)
    print("Output from full forward pass shape:", out_full.shape)
    print("Output from full forward pass sample (first token):")
    print(out_full[0, 0])
    print("Forward pass output matches manual?", torch.allclose(out, out_full))
    print()

def test_decoder():
    print("=== Test Decoder ===")
    d_model = 8
    d_ff = 16
    h = 2
    dropout = 0.0
    batch_size = 2
    src_seq_len = 5
    tgt_seq_len = 4
    num_layers = 2

    # Build list of DecoderBlocks
    layers = nn.ModuleList([
        DecoderBlock(
            d_model=d_model,
            self_attention_block=MultiHeadAttentionBlock(d_model=d_model, num_heads=h, dropout=dropout),
            cross_attention_block=MultiHeadAttentionBlock(d_model=d_model, num_heads=h, dropout=dropout),
            feed_forward_block=FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout),
            dropout=dropout
        )
        for _ in range(num_layers)
    ])

    decoder = Decoder(d_model=d_model, layers=layers)

    # Input
    x = torch.randn(batch_size, tgt_seq_len, d_model) # random tensor of shape (2, 4, 8)
    enc_output = torch.randn(batch_size, src_seq_len, d_model) # random tensor of shape (2, 5, 8)
    tgt_mask = None  # No mask for this test
    src_mask = None  # No mask for this test

    print("Decoder Input x shape:", x.shape)
    print("Encoder Output shape:", enc_output.shape)
    print("Decoder Input sample (first token):")
    print(x[0, 0])
    print()

    # Manual forward pass through all layers
    x_manual = x
    for i, layer in enumerate(decoder.layers):
        print(f"--- Decoder Layer {i+1} ---")
        x_manual = layer(x_manual, enc_output, src_mask, tgt_mask)
        print("Output shape after this layer:", x_manual.shape)
        print("Output sample (first token):")
        print(x_manual[0, 0])
        print()

    x_manual_norm = decoder.layer_norm(x_manual)
    print("After final LayerNorm shape:", x_manual_norm.shape)
    print("After final LayerNorm sample (first token):")
    print(x_manual_norm[0, 0])
    print()

    # Decoder forward pass
    decoder_out = decoder(x, enc_output, src_mask, tgt_mask)
    print("Output from Decoder forward pass shape:", decoder_out.shape)
    print("Output from Decoder forward pass sample (first token):")
    print(decoder_out[0, 0])
    print()

    # Compare manual and forward pass
    print("Forward pass output matches manual?", torch.allclose(x_manual_norm, decoder_out))
    print()

def test_projection_layer():
    print("=== Test ProjectionLayer ===")
    d_model = 8
    vocab_size = 10
    batch_size = 2
    seq_len = 4

    proj = ProjectionLayer(d_model=d_model, vocab_size=vocab_size)

    # Input tensor (fake decoder output)
    x = torch.randn(batch_size, seq_len, d_model) # random tensor of shape (2, 4, 8)

    print("Input x shape:", x.shape)  # (2, 4, 8)
    print("Vocab size:", vocab_size)
    print("Input sample (first token vector):")
    print(x[0, 0])
    print()

    # Forward pass
    out = proj(x) # shape: (2, 4, vocab_size)

    print("Output shape:", out.shape)  # (2, 4, vocab_size)
    print("Output sample (first token logits):")
    print(out[0, 0])
    print()

    # Meaning Check
    probs = torch.softmax(out, dim=-1)
    print("Probabilities shape (after softmax):", probs.shape)  # (2, 4, vocab_size)
    print("Probabilities sample (first token):")
    print(probs[0, 0])
    print("Sum of probabilities for first token (should be 1):", probs[0, 0].sum().item())
    print()

def test_transformer():
    print("=== Test Transformer (end-to-end) ===")

    src_vocab_size = 11 # source vocabulary size
    tgt_vocab_size = 13 # target vocabulary size
    src_seq_len = 5 # source sequence length
    tgt_seq_len = 4 # target sequence length
    d_model = 8 # model dimension
    N = 2 # number of layers
    h = 2 # number of heads
    d_ff = 16 # feedforward dimension
    dropout = 0.0 # set 0.0 for deterministic behavior
    batch_size = 2

    # Build a tiny Transformer
    transformer = Transformer.build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len,
        d_model=d_model,
        num_heads=h,
        d_ff=d_ff,
        num_layers=N,
        dropout=dropout
    )

    # Input: fake random token ids for source and target
    src_tokens = torch.randint(0, src_vocab_size, (batch_size, src_seq_len)) # shape: (2, 5)
    tgt_tokens = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len)) # shape: (2, 4)

    src_mask = None # no mask for this test
    tgt_mask = None # no mask for this test

    print("Source tokens shape:", src_tokens.shape) # (2, 5)
    print("Source tokens:\n", src_tokens)
    print()
    print("Target tokens shape:", tgt_tokens.shape) # (2, 4)
    print("Target tokens:\n", tgt_tokens)
    print()

    # Encoder path step by step
    print("--- Encoder Path ---")

    src_embed = transformer.src_embed(src_tokens) # (2, 5, d_model)
    print("After InputEmbedding shape:", src_embed.shape)
    print("Sample (first token vector, first batch):")
    print(src_embed[0, 0])
    print()

    src_emb_pos = transformer.src_pos(src_embed) # (2, 5, d_model)
    print("After PositionalEncoding shape:", src_emb_pos.shape)
    print("Sample (first token vector, first batch):")
    print(src_emb_pos[0, 0])
    print()

    enc_output = transformer.encoder(src_emb_pos, src_mask) # (2, 5, d_model)
    print("After Encoder shape:", enc_output.shape)
    print("Sample (first token vector, first batch):")
    print(enc_output[0, 0])
    print()

    # with encode method
    enc_output_via_encode = transformer.encode(src_tokens, src_mask)
    print("Encoder output via encode() shape:", enc_output_via_encode.shape)
    print("Sample (first token vector, first batch):")
    print(enc_output_via_encode[0, 0])
    print()

    print("Encoder output via encode() matches manual?", torch.allclose(enc_output, enc_output_via_encode))
    print()

    # Decoder path step by step
    print("--- Decoder Path ---")

    tgt_embed = transformer.tgt_embed(tgt_tokens) # (2, 4, d_model)
    print("After InputEmbedding shape:", tgt_embed.shape)
    print("Sample (first token vector, first batch):")
    print(tgt_embed[0, 0])
    print()

    tgt_emb_pos = transformer.tgt_pos(tgt_embed) # (2, 4, d_model)
    print("After PositionalEncoding shape:", tgt_emb_pos.shape)
    print("Sample (first token vector, first batch):")
    print(tgt_emb_pos[0, 0])
    print()

    dec_output = transformer.decoder(tgt_emb_pos, enc_output, src_mask, tgt_mask) # (2, 4, d_model)
    print("After Decoder shape:", dec_output.shape)
    print("Sample (first token vector, first batch):")
    print(dec_output[0, 0])
    print()

    # with decode method
    dec_output_via_decode = transformer.decode(tgt_tokens, enc_output, src_mask, tgt_mask)
    print("Decoder output via decode() shape:", dec_output_via_decode.shape)
    print("Sample (first token vector, first batch):")
    print(dec_output_via_decode[0, 0])
    print()

    print("Decoder output via decode() matches manual?", torch.allclose(dec_output, dec_output_via_decode))
    print()

    # Projection layer
    proj_output = transformer.projection_layer(dec_output) # (2, 4, tgt_vocab_size)
    print("After ProjectionLayer shape:", proj_output.shape)
    print("Sample (first token logits, first batch):")
    print(proj_output[0, 0])
    print()

    # Full forward pass
    full_output = transformer.project(dec_output) # (2, 4, tgt_vocab_size)
    print("Output from full Transformer forward pass shape:", full_output.shape)
    print("Sample (first token logits, first batch):")
    print(full_output[0, 0])
    print()

    print("Full forward pass output matches manual?", torch.allclose(proj_output, full_output))
    print()

    # View probabilities for one token
    probs = torch.softmax(full_output, dim=-1)
    print("Probabilities shape (after softmax):", probs.shape)  # (2, 4, tgt_vocab_size)
    print("Probabilities sample (first token, first batch):")
    print(probs[0, 0])
    print("Sum of probabilities for first token (should be 1):", probs[0, 0].sum().item())
    print()

def inspect_pt_checkpoint(ckpt_path):

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print("Checkpoint path:", ckpt_path)
    print("Keys:", ckpt.keys())

    if "epoch" in ckpt:
        print("Epoch:", ckpt["epoch"])

    if "global_step" in ckpt:
        print("Global step:", ckpt["global_step"])

    # if "model_weights" in ckpt:
    #     print("Number of model tensors:", len(ckpt["model_weights"]))

    if "optimizer_state" in ckpt:
        print("Optimizer keys:", ckpt["optimizer_state"].keys())

    # if "total_params" in ckpt:
    #     print("Total params:", ckpt["total_params"])

    if "trainable_params" in ckpt:
        print("Trainable params:", ckpt["trainable_params"])

    if "epoch_time_sec" in ckpt:
        print("Epoch time (sec):", ckpt["epoch_time_sec"])

    if "total_training_time_sec" in ckpt:
        print("Total training time (sec):", ckpt["total_training_time_sec"])


if __name__ == "__main__":
    # Uncomment to run individual tests

    # test_input_embedding()
    # test_input_embeddings_with_vocab()
    # test_positional_encoding()
    # test_embeddings_plus_positional()
    # test_layer_normalization()
    # test_feedforward_block()
    # test_residual_connection()
    # test_multihead_attention_block()
    # test_encoder_block()
    # test_encoder()
    # test_decoder_block()
    # test_decoder()
    # test_projection_layer()
    # test_transformer()
    inspect_pt_checkpoint("transformer_weights/tmodel_02.pt")
