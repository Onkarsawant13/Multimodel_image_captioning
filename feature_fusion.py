"""
Feature Fusion Layer — Module 4 of Image Captioning Architecture
=================================================================
Fuses outputs from all three upstream modules into a single unified
sequence of tokens that the Transformer Decoder will attend to.

Inputs (from previous modules):
  • ObjectModel    → object_features  : (B, 16, 512)   object tokens
  • SceneModel     → scene_features   : (B, 50, 512)   scene tokens (CLS + 49 patches)
  • RelationModule → relation_features: (B, 16, 512)   relation-enriched tokens

BACKGROUND DETECTION FIX:
  YOLO only detects foreground objects (people, cars, dogs etc.)
  It cannot detect background environments: sky, beach, snow, forest, indoor rooms.
  Fix: inject CLIP's CLS token as a "background token" at position 0 of the
  object stream. The CLS token captures the overall scene type from CLIP which
  was trained on image-text pairs — it knows "beach", "kitchen", "mountain" etc.
  This lets the decoder generate captions like:
    "a man walking on a beach"  ← beach from background token, man from YOLO
    "children playing in the snow"  ← snow from background token

Fusion strategy — THREE steps:
  1. Stream Gating   : learned gates decide how much each stream contributes
                       per token (adaptive per image)
  2. Cross-Attention : object/relation tokens attend to scene tokens
  3. Transformer     : 2-layer transformer over the full 82-token sequence

Output fed into Transformer Decoder:
  fused_features : (B, 82, 512)   — 16 obj + 16 rel + 50 scene = 82 tokens
  fused_mask     : (B, 82)        — True = padding

Dependencies:
    pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
# Config  (must match all upstream modules)
# ──────────────────────────────────────────────────────────────────
D_MODEL      = 512
MAX_OBJECTS  = 16
NUM_SCENE    = 50    # 49 patches + 1 CLS
SEQ_LEN      = MAX_OBJECTS + MAX_OBJECTS + NUM_SCENE   # 82 tokens


# ──────────────────────────────────────────────────────────────────
# 1. Stream Gate
# ──────────────────────────────────────────────────────────────────
class StreamGate(nn.Module):
    """
    Learned per-token gating between object and relation streams.
    gate_fc input: d_model * 2 (obj concat rel)
    """
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, obj_feats: torch.Tensor, rel_feats: torch.Tensor) -> torch.Tensor:
        gates = self.gate_fc(torch.cat([obj_feats, rel_feats], dim=-1))  # (B, N, 2)
        g_obj = gates[..., 0:1]
        g_rel = gates[..., 1:2]
        return self.layer_norm(g_obj * obj_feats + g_rel * rel_feats)


# ──────────────────────────────────────────────────────────────────
# 2. Scene-Aware Cross-Attention
# ──────────────────────────────────────────────────────────────────
class SceneAwareCrossAttention(nn.Module):
    """
    Objects/relations attend to scene tokens → become scene-aware.
    Query = object tokens, Key/Value = scene tokens.
    """
    def __init__(self, d_model: int = D_MODEL, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=0.1, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(0.1)

    def forward(self, obj_feats, scene_feats, obj_mask):
        attn_out, _ = self.cross_attn(
            query=obj_feats, key=scene_feats, value=scene_feats,
            key_padding_mask=None,
        )
        out = self.layer_norm(obj_feats + self.dropout(attn_out))
        return out.masked_fill(obj_mask.unsqueeze(-1), 0.0)


# ──────────────────────────────────────────────────────────────────
# 3. Fusion Transformer
# ──────────────────────────────────────────────────────────────────
class FusionTransformer(nn.Module):
    """2-layer transformer encoder for global cross-stream interaction."""
    def __init__(self, d_model=D_MODEL, num_heads=8, num_layers=2,
                 dim_ff=2048, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, mask):
        return self.transformer(x, src_key_padding_mask=mask)


# ──────────────────────────────────────────────────────────────────
# 4. FeatureFusionLayer  (top-level)
# ──────────────────────────────────────────────────────────────────
class FeatureFusionLayer(nn.Module):
    """
    Stable v1 fusion architecture + background token injection.

    Background token: CLIP CLS feature projected into object stream space,
    injected at position 0 of the object and relation streams so the decoder
    always knows what kind of scene/environment is in the image.
    """

    def __init__(self, d_model=D_MODEL, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()

        # Stream type embeddings (BERT-style segment IDs)
        # 0 = object, 1 = relation, 2 = scene
        self.stream_embed = nn.Embedding(3, d_model)

        # Background token projection: CLIP CLS → object stream space
        self.bg_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        # Fusion components
        self.stream_gate        = StreamGate(d_model)
        self.cross_attn_obj     = SceneAwareCrossAttention(d_model, num_heads)
        self.cross_attn_rel     = SceneAwareCrossAttention(d_model, num_heads)
        self.fusion_transformer = FusionTransformer(d_model, num_heads, num_layers)

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, object_model_out, scene_model_out, relation_module_out):
        """
        Args:
            object_model_out    : {"object_features":(B,16,512), "object_mask":(B,16)}
            scene_model_out     : {"scene_features":(B,50,512),  "scene_mask":(B,50),
                                   "cls_feature":(B,512)}
            relation_module_out : {"relation_features":(B,16,512), "relation_mask":(B,16)}

        Returns:
            {"fused_features":(B,82,512), "fused_mask":(B,82), "seq_layout":dict}
        """
        obj_feats   = object_model_out["object_features"]       # (B, 16, 512)
        obj_mask    = object_model_out["object_mask"]           # (B, 16)
        scene_feats = scene_model_out["scene_features"]         # (B, 50, 512)
        scene_mask  = scene_model_out["scene_mask"]             # (B, 50)
        cls_feat    = scene_model_out["cls_feature"]            # (B, 512)
        rel_feats   = relation_module_out["relation_features"]  # (B, 16, 512)
        rel_mask    = relation_module_out["relation_mask"]      # (B, 16)

        B      = obj_feats.shape[0]
        device = obj_feats.device

        # ── BACKGROUND TOKEN INJECTION ──
        # Project CLIP CLS token into object/relation stream space
        bg_token = self.bg_proj(cls_feat)   # (B, 512)

        # Clone to avoid in-place modification of upstream tensors
        obj_feats = obj_feats.clone()
        obj_mask  = obj_mask.clone()
        rel_feats = rel_feats.clone()
        rel_mask  = rel_mask.clone()

        # Inject at position 0 of both streams, mark as valid (not padding)
        obj_feats[:, 0, :] = bg_token
        obj_mask[:, 0]     = False
        rel_feats[:, 0, :] = bg_token
        rel_mask[:, 0]     = False

        # ── Step 1: Stream Gating ──
        gated_obj = self.stream_gate(obj_feats, rel_feats)      # (B, 16, 512)

        # ── Step 2: Cross-Attention ──
        scene_aware_obj = self.cross_attn_obj(gated_obj, scene_feats, obj_mask)
        scene_aware_rel = self.cross_attn_rel(rel_feats, scene_feats, rel_mask)

        # ── Step 3: Stream type embeddings ──
        obj_ids   = torch.zeros(B, MAX_OBJECTS, dtype=torch.long, device=device)
        rel_ids   = torch.ones (B, MAX_OBJECTS, dtype=torch.long, device=device)
        scene_ids = torch.full ((B, NUM_SCENE),  2, dtype=torch.long, device=device)

        scene_aware_obj = scene_aware_obj + self.stream_embed(obj_ids)
        scene_aware_rel = scene_aware_rel + self.stream_embed(rel_ids)
        scene_feats_emb = scene_feats     + self.stream_embed(scene_ids)

        # ── Step 4: Concatenate → 82 tokens ──
        fused_seq  = torch.cat([scene_aware_obj, scene_aware_rel, scene_feats_emb], dim=1)
        fused_mask = torch.cat([obj_mask, rel_mask, scene_mask], dim=1)

        # ── Step 5: Fusion Transformer ──
        fused_seq = self.dropout(fused_seq)
        fused_out = self.fusion_transformer(fused_seq, fused_mask)
        fused_out = self.layer_norm(fused_out)

        # Zero out padding positions
        fused_out = fused_out.masked_fill(fused_mask.unsqueeze(-1), 0.0)

        return {
            "fused_features": fused_out,
            "fused_mask":     fused_mask,
            "seq_layout": {
                "object_start":   0,
                "object_end":     MAX_OBJECTS,
                "relation_start": MAX_OBJECTS,
                "relation_end":   MAX_OBJECTS * 2,
                "scene_start":    MAX_OBJECTS * 2,
                "scene_end":      MAX_OBJECTS * 2 + NUM_SCENE,
            }
        }


# ──────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)
    B = 2

    obj_out = {
        "object_features": torch.randn(B, MAX_OBJECTS, D_MODEL).to(device),
        "object_mask":     torch.zeros(B, MAX_OBJECTS, dtype=torch.bool).to(device),
    }
    obj_out["object_mask"][:, 10:] = True

    scene_out = {
        "scene_features": torch.randn(B, NUM_SCENE, D_MODEL).to(device),
        "scene_mask":     torch.zeros(B, NUM_SCENE, dtype=torch.bool).to(device),
        "cls_feature":    torch.randn(B, D_MODEL).to(device),
    }

    rel_out = {
        "relation_features": torch.randn(B, MAX_OBJECTS, D_MODEL).to(device),
        "relation_mask":     torch.zeros(B, MAX_OBJECTS, dtype=torch.bool).to(device),
    }
    rel_out["relation_mask"][:, 10:] = True

    model = FeatureFusionLayer(d_model=512, num_heads=8, num_layers=2).to(device)
    model.eval()

    with torch.no_grad():
        out = model(obj_out, scene_out, rel_out)

    print("fused_features :", out["fused_features"].shape)   # (2, 82, 512)
    print("fused_mask     :", out["fused_mask"].shape)

    layout = out["seq_layout"]
    print(f"Object   tokens : {layout['object_start']}–{layout['object_end']-1}  (pos 0 = background)")
    print(f"Relation tokens : {layout['relation_start']}–{layout['relation_end']-1}")
    print(f"Scene    tokens : {layout['scene_start']}–{layout['scene_end']-1}")

    bg_valid = not out["fused_mask"][0, 0].item()
    print(f"Background token valid : {bg_valid}")
    print(f"All finite             : {out['fused_features'].isfinite().all().item()}")

    n_keys = len(model.state_dict())
    print(f"State dict keys        : {n_keys}  (target: ~29)")
    print("\n✓ FeatureFusionLayer v1+background working correctly!")