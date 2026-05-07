"""
Transformer Decoder + Caption Head — Module 5 of Image Captioning Architecture
===============================================================================
Uses PyTorch's built-in nn.TransformerDecoder for stable, fast training.

Architecture:
    Token Embedding + Positional Encoding
          ↓
    4 × nn.TransformerDecoderLayer
        - Masked Self-Attention  (causal)
        - Cross-Attention        (attends to 82 fused tokens)
        - Feed-Forward
          ↓
    Linear → vocab logits
          ↓
    Caption tokens (beam search or greedy)

Why built-in decoder over custom coverage decoder:
    - PyTorch's implementation is heavily optimised (fused ops, Flash Attention)
    - More stable training — fewer custom ops to go wrong
    - Faster convergence — reaches good BLEU-4 in fewer epochs
    - Coverage can be added later as a post-hoc loss term without
      replacing the entire decoder architecture

Dependencies:
    pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
D_MODEL     = 512
VOCAB_SIZE  = 10000
MAX_SEQ_LEN = 50
NUM_LAYERS  = 4
NUM_HEADS   = 8
DIM_FF      = 2048
DROPOUT     = 0.1
PAD_ID      = 0
BOS_ID      = 1
EOS_ID      = 2


# ──────────────────────────────────────────────
# 1. Positional Encoding
# ──────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, max_len=MAX_SEQ_LEN, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ──────────────────────────────────────────────
# 2. CaptionDecoder — built-in TransformerDecoder
# ──────────────────────────────────────────────
class CaptionDecoder(nn.Module):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, tgt, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask=None):
        return self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )


# ──────────────────────────────────────────────
# 3. ImageCaptionModel
# ──────────────────────────────────────────────
class ImageCaptionModel(nn.Module):
    """
    Complete caption model:
        fused visual features (B, 82, 512)
            → TransformerDecoder (4 layers)
            → Linear(512, vocab_size)
            → caption tokens

    Training : teacher forcing + label smoothing
    Inference : beam search (k=5) or greedy
    """

    def __init__(
        self,
        vocab_size:  int   = VOCAB_SIZE,
        d_model:     int   = D_MODEL,
        num_layers:  int   = NUM_LAYERS,
        num_heads:   int   = NUM_HEADS,
        dim_ff:      int   = DIM_FF,
        max_seq_len: int   = MAX_SEQ_LEN,
        dropout:     float = DROPOUT,
        pad_id:      int   = PAD_ID,
        bos_id:      int   = BOS_ID,
        eos_id:      int   = EOS_ID,
    ):
        super().__init__()
        self.d_model     = d_model
        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id      = pad_id
        self.bos_id      = bos_id
        self.eos_id      = eos_id

        # Token embedding + positional encoding
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc     = PositionalEncoding(d_model, max_seq_len, dropout)
        self.embed_scale = math.sqrt(d_model)

        # Decoder
        self.decoder = CaptionDecoder(
            d_model, num_heads, num_layers, dim_ff, dropout
        )

        # Output projection → vocabulary (weight tied with embedding)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1,
        )

    # ── Training forward ─────────────────────────────────────────────
    def forward(self, fused_out, caption_ids):
        memory      = fused_out["fused_features"]
        memory_mask = fused_out["fused_mask"]
        B, T        = caption_ids.shape
        device      = caption_ids.device

        tgt     = self.token_embed(caption_ids) * self.embed_scale
        tgt     = self.pos_enc(tgt)
        causal  = self._make_causal_mask(T, device)
        pad_msk = (caption_ids == self.pad_id)

        decoded = self.decoder(tgt, memory, causal, memory_mask, pad_msk)
        return self.output_proj(decoded)

    # ── Training loss ─────────────────────────────────────────────────
    def compute_loss(self, fused_out, caption_ids):
        """
        Cross-entropy loss with label smoothing.
        caption_ids: (B, T+1) — [BOS, w1, ..., wN, EOS, PAD...]
        """
        inp    = caption_ids[:, :-1]   # (B, T) — input tokens
        target = caption_ids[:, 1:]    # (B, T) — prediction targets

        logits = self.forward(fused_out, inp)   # (B, T, vocab_size)

        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target.reshape(-1),
            ignore_index=self.pad_id,
            label_smoothing=0.1,
        )

    # ── Greedy decoding ───────────────────────────────────────────────
    @torch.no_grad()
    def generate_greedy(self, fused_out, device):
        memory      = fused_out["fused_features"].to(device)
        memory_mask = fused_out["fused_mask"].to(device)
        B           = memory.shape[0]

        tokens   = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(self.max_seq_len - 1):
            T_cur   = tokens.shape[1]
            tgt     = self.token_embed(tokens) * self.embed_scale
            tgt     = self.pos_enc(tgt)
            causal  = self._make_causal_mask(T_cur, device)

            decoded    = self.decoder(tgt, memory, causal, memory_mask)
            next_token = self.output_proj(decoded[:, -1, :]).argmax(
                dim=-1, keepdim=True
            )
            next_token = next_token.masked_fill(finished.unsqueeze(1), self.pad_id)
            tokens     = torch.cat([tokens, next_token], dim=1)
            finished   = finished | (next_token.squeeze(1) == self.eos_id)
            if finished.all():
                break

        return tokens

    # ── Beam search ───────────────────────────────────────────────────
    @torch.no_grad()
    def generate_beam(self, fused_out, device, beam_size=5, length_penalty=0.7):
        memory      = fused_out["fused_features"].to(device)
        memory_mask = fused_out["fused_mask"].to(device)
        B           = memory.shape[0]
        results     = []

        for b in range(B):
            mem      = memory[b:b+1]
            mem_mask = memory_mask[b:b+1]
            beams    = [(0.0, [self.bos_id])]
            completed = []

            for _ in range(self.max_seq_len - 1):
                all_candidates = []
                for score, toks in beams:
                    if toks[-1] == self.eos_id:
                        completed.append((score, toks))
                        continue

                    t      = torch.tensor(toks, dtype=torch.long,
                                          device=device).unsqueeze(0)
                    tgt    = self.token_embed(t) * self.embed_scale
                    tgt    = self.pos_enc(tgt)
                    causal = self._make_causal_mask(t.shape[1], device)

                    decoded   = self.decoder(tgt, mem, causal, mem_mask)
                    log_probs = F.log_softmax(
                        self.output_proj(decoded[:, -1, :]), dim=-1
                    )
                    topk_lp, topk_ids = log_probs[0].topk(beam_size)

                    for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                        all_candidates.append((score + lp, toks + [tid]))

                if not all_candidates:
                    break

                all_candidates.sort(
                    key=lambda x: x[0] / (len(x[1]) ** length_penalty),
                    reverse=True,
                )
                beams = all_candidates[:beam_size]

                if all(t[-1] == self.eos_id for _, t in beams):
                    completed.extend(beams)
                    break

            if not completed:
                completed = beams

            best = max(
                completed,
                key=lambda x: x[0] / (len(x[1]) ** length_penalty),
            )
            results.append(best[1])

        return results


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(42)

    fused_out = {
        "fused_features": torch.randn(2, 82, D_MODEL).to(device),
        "fused_mask":     torch.zeros(2, 82, dtype=torch.bool).to(device),
    }
    fused_out["fused_mask"][:, 10:16] = True

    model = ImageCaptionModel(vocab_size=10000).to(device)
    model.eval()

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    model.train()
    caps = torch.randint(3, 10000, (2, 16)).to(device)
    caps[:, 0] = BOS_ID; caps[:, -1] = EOS_ID
    loss = model.compute_loss(fused_out, caps)
    print(f"Loss: {loss.item():.4f}")

    # Beam search
    model.eval()
    beam = model.generate_beam(fused_out, device, beam_size=3)
    print(f"Beam lengths: {[len(b) for b in beam]}")
    print("OK ✓")