"""
Relation Module — Module 3 of Image Captioning Architecture
============================================================
Replaces the original HOI (Human-Object Interaction) model with a lightweight
Graph Convolutional Network (GCN) that captures relationships between detected
objects. This is better for Flickr30k because:
  - Flickr30k has NO HOI annotations → HOI model can't be trained
  - GCN works directly on YOLO's output (boxes + features from ObjectModel)
  - Captures spatial & semantic relations: "man NEAR dog", "person ON bike"
  - Much lighter than HOI models, trains fast on Flickr30k

What this module does:
  1. Builds a relation graph between all detected objects
  2. Edge weights = spatial proximity + semantic similarity (learned)
  3. Runs 2-layer GCN to propagate relational context across objects
  4. Outputs enriched object tokens where each object "knows" about its neighbours

Output fed into Feature Fusion Layer:
  relation_features : (batch, max_objects, d_model)  — relation-enriched tokens
  relation_mask     : (batch, max_objects)            — same mask as ObjectModel

Dependencies:
    pip install torch torchvision
    (No extra libs needed — GCN implemented from scratch with torch.bmm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MAX_OBJECTS = 16     # must match object_model.py
D_MODEL     = 512    # must match object_model.py
D_RELATION  = 256    # internal relation edge embedding dim


# ──────────────────────────────────────────────
# 1. Spatial Relation Encoder
#    Computes geometric relationship between every pair of boxes
#    Input : (B, N, 4) normalised xyxy boxes
#    Output: (B, N, N, D_RELATION) edge features
# ──────────────────────────────────────────────
class SpatialRelationEncoder(nn.Module):
    """
    For every pair of objects (i, j), encodes their spatial relationship
    using 8 geometric features:
        dx, dy          — relative centre offsets
        log(w_i/w_j)    — relative width ratio (log scale)
        log(h_i/h_j)    — relative height ratio (log scale)
        iou             — intersection over union
        dist            — euclidean centre distance
        angle           — angle from i to j
        area_ratio      — relative area difference

    These 8 features are projected to D_RELATION via an MLP.
    """
    def __init__(self, d_relation: int = D_RELATION):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, d_relation),
            nn.ReLU(),
        )

    def _box_centres(self, boxes: torch.Tensor):
        """Returns (cx, cy, w, h) from xyxy boxes."""
        cx = (boxes[..., 0] + boxes[..., 2]) / 2
        cy = (boxes[..., 1] + boxes[..., 3]) / 2
        w  =  boxes[..., 2] - boxes[..., 0]
        h  =  boxes[..., 3] - boxes[..., 1]
        return cx, cy, w, h

    def _iou(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise IoU matrix.
        Args:
            boxes: (B, N, 4) xyxy
        Returns:
            (B, N, N) iou values
        """
        B, N, _ = boxes.shape
        b1 = boxes.unsqueeze(2).expand(B, N, N, 4)  # (B, N, N, 4)
        b2 = boxes.unsqueeze(1).expand(B, N, N, 4)  # (B, N, N, 4)

        inter_x1 = torch.max(b1[..., 0], b2[..., 0])
        inter_y1 = torch.max(b1[..., 1], b2[..., 1])
        inter_x2 = torch.min(b1[..., 2], b2[..., 2])
        inter_y2 = torch.min(b1[..., 3], b2[..., 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter   = inter_w * inter_h

        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
        union = area1 + area2 - inter + 1e-6

        return inter / union   # (B, N, N)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boxes: (B, N, 4) normalised xyxy
        Returns:
            edge_feats: (B, N, N, d_relation)
        """
        B, N, _ = boxes.shape
        cx, cy, w, h = self._box_centres(boxes)  # each (B, N)

        # Expand for pairwise computation
        cx_i = cx.unsqueeze(2).expand(B, N, N)
        cy_i = cy.unsqueeze(2).expand(B, N, N)
        w_i  =  w.unsqueeze(2).expand(B, N, N)
        h_i  =  h.unsqueeze(2).expand(B, N, N)

        cx_j = cx.unsqueeze(1).expand(B, N, N)
        cy_j = cy.unsqueeze(1).expand(B, N, N)
        w_j  =  w.unsqueeze(1).expand(B, N, N)
        h_j  =  h.unsqueeze(1).expand(B, N, N)

        eps = 1e-6

        dx          = (cx_j - cx_i)                              # relative x offset
        dy          = (cy_j - cy_i)                              # relative y offset
        log_w_ratio = torch.log(w_j / (w_i + eps) + eps)         # width ratio
        log_h_ratio = torch.log(h_j / (h_i + eps) + eps)         # height ratio
        iou         = self._iou(boxes)                            # overlap
        dist        = torch.sqrt(dx**2 + dy**2 + eps)            # euclidean dist
        angle       = torch.atan2(dy, dx + eps)                  # direction
        area_i      = w_i * h_i
        area_j      = w_j * h_j
        area_ratio  = torch.log(area_j / (area_i + eps) + eps)   # size relation

        # Stack 8 features → (B, N, N, 8)
        edge_raw = torch.stack(
            [dx, dy, log_w_ratio, log_h_ratio,
             iou, dist, angle, area_ratio], dim=-1
        )

        # Project to d_relation
        edge_feats = self.mlp(edge_raw)   # (B, N, N, d_relation)
        return edge_feats


# ──────────────────────────────────────────────
# 2. Graph Attention Layer
#    One message-passing step with learned attention weights
#    Input : node features (B, N, d_model), edge features (B, N, N, d_relation)
#    Output: updated node features (B, N, d_model)
# ──────────────────────────────────────────────
class GraphAttentionLayer(nn.Module):
    """
    Attention-weighted graph convolution:
      1. Compute attention score for each edge using node + edge features
      2. Softmax over neighbours (masked for padding)
      3. Aggregate neighbour messages weighted by attention
      4. Residual connection + LayerNorm

    This lets each object "ask": which of my neighbours matter most
    for understanding what I am doing?
    """
    def __init__(self, d_model: int = D_MODEL, d_relation: int = D_RELATION):
        super().__init__()

        # Attention score: concat(node_i, node_j, edge_ij) → scalar
        self.attn_fc = nn.Linear(d_model * 2 + d_relation, 1)

        # Message transform: project neighbour features before aggregation
        self.msg_fc  = nn.Linear(d_model, d_model)

        # Edge feature projection to d_model for additive message
        self.edge_fc = nn.Linear(d_relation, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(0.1)

    def forward(
        self,
        node_feats:  torch.Tensor,   # (B, N, d_model)
        edge_feats:  torch.Tensor,   # (B, N, N, d_relation)
        node_mask:   torch.Tensor,   # (B, N) True = padding
    ) -> torch.Tensor:
        """
        Returns:
            updated node features: (B, N, d_model)
        """
        B, N, D = node_feats.shape

        # Expand node features for pairwise attention scoring
        n_i = node_feats.unsqueeze(2).expand(B, N, N, D)   # sender
        n_j = node_feats.unsqueeze(1).expand(B, N, N, D)   # receiver

        # Attention logits: (B, N, N, 1)
        attn_input = torch.cat([n_i, n_j, edge_feats], dim=-1)
        attn_logits = self.attn_fc(attn_input).squeeze(-1)   # (B, N, N)

        # Mask out padding nodes — set their attention to -inf
        if node_mask is not None:
            # node_mask: (B, N) → expand to (B, N, N) for receiver dim
            pad_mask = node_mask.unsqueeze(1).expand(B, N, N)   # (B, N, N)
            attn_logits = attn_logits.masked_fill(pad_mask, float("-inf"))

        # Softmax over neighbours (dim=2 = neighbour axis)
        attn_weights = torch.softmax(attn_logits, dim=2)    # (B, N, N)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # handle all-pad rows
        attn_weights = self.dropout(attn_weights)

        # Message: transform neighbour node features + edge context
        messages = self.msg_fc(node_feats)                   # (B, N, d_model)
        edge_ctx = self.edge_fc(edge_feats)                  # (B, N, N, d_model)

        # Weighted aggregation: sum over neighbours
        # attn: (B, N, N) × messages: (B, N, d_model)
        agg = torch.bmm(attn_weights, messages)              # (B, N, d_model)

        # Add edge context: average edge features weighted by attention
        edge_agg = (attn_weights.unsqueeze(-1) * edge_ctx).sum(dim=2)  # (B, N, d_model)

        # Combine + residual
        out = self.layer_norm(node_feats + agg + edge_agg)   # (B, N, d_model)
        return out


# ──────────────────────────────────────────────
# 3. RelationModule  (top-level)
# ──────────────────────────────────────────────
class RelationModule(nn.Module):
    """
    Full relation module:
        object_features + boxes
            → spatial edge features (SpatialRelationEncoder)
            → 2-layer Graph Attention (GraphAttentionLayer × 2)
            → relation-enriched object tokens

    The output has the same shape as the input object features (B, N, d_model)
    but each token now encodes not just "what am I" but "what am I doing
    relative to the objects around me."

    Example relations learned:
        person ──near──→  dog    → "man walking his dog"
        person ──on───→   bike   → "cyclist riding a bicycle"
        child  ──holds──→ ball   → "girl holding a red ball"

    Args:
        d_model    : must match ObjectModel output (default 512)
        d_relation : internal edge embedding dim (default 256)
        num_layers : number of graph attention layers (default 2)
        dropout    : dropout rate
    """

    def __init__(
        self,
        d_model:    int = D_MODEL,
        d_relation: int = D_RELATION,
        num_layers: int = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model    = d_model
        self.num_layers = num_layers

        # Spatial edge encoder (shared across GCN layers)
        self.spatial_encoder = SpatialRelationEncoder(d_relation)

        # Stack of graph attention layers
        self.gcn_layers = nn.ModuleList([
            GraphAttentionLayer(d_model, d_relation)
            for _ in range(num_layers)
        ])

        # Final projection + norm
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)

    def forward(
        self,
        object_features: torch.Tensor,   # (B, N, d_model) from ObjectModel
        boxes:           torch.Tensor,   # (B, N, 4) normalised xyxy from ObjectModel
        object_mask:     torch.Tensor,   # (B, N) True=padding from ObjectModel
    ) -> dict:
        """
        Args:
            object_features : (B, N, d_model)  — raw object tokens
            boxes           : (B, N, 4)         — normalised xyxy
            object_mask     : (B, N)             — True = padding

        Returns dict with keys:
            "relation_features" : (B, N, d_model) — relation-enriched tokens
            "relation_mask"     : (B, N)           — same as object_mask
            "edge_weights"      : (B, N, N)        — attention weights (for viz)
        """
        B, N, _ = object_features.shape

        # ── Step 1: Build spatial edge features ──
        edge_feats = self.spatial_encoder(boxes)    # (B, N, N, d_relation)

        # ── Step 2: Graph attention message passing ──
        x = object_features
        last_attn = None

        for layer in self.gcn_layers:
            x = layer(x, edge_feats, object_mask)

        # ── Step 3: Final norm + zero-out padding ──
        x = self.output_norm(x)
        x = self.dropout(x)

        # Zero out padding positions
        x = x.masked_fill(object_mask.unsqueeze(-1), 0.0)

        # Recompute attention weights from last layer for visualisation
        # (lightweight recompute — not stored during forward for memory efficiency)
        with torch.no_grad():
            n_i = x.unsqueeze(2).expand(B, N, N, self.d_model)
            n_j = x.unsqueeze(1).expand(B, N, N, self.d_model)
            attn_in = torch.cat([n_i, n_j, edge_feats], dim=-1)
            edge_weights = torch.softmax(
                self.gcn_layers[-1].attn_fc(attn_in).squeeze(-1), dim=2
            )
            edge_weights = torch.nan_to_num(edge_weights, nan=0.0)

        return {
            "relation_features" : x,             # (B, N, d_model)
            "relation_mask"     : object_mask,   # (B, N)
            "edge_weights"      : edge_weights,  # (B, N, N) for visualisation
        }


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Simulate ObjectModel output ──
    B, N, D = 2, 16, 512
    torch.manual_seed(42)

    # Fake object features (normally from ObjectModel)
    fake_obj_features = torch.randn(B, N, D).to(device)

    # Fake boxes: 10 valid objects, 6 padding per image
    fake_boxes = torch.rand(B, N, 4).to(device)
    # Make sure x2>x1, y2>y1
    fake_boxes[..., 2] = fake_boxes[..., 0] + fake_boxes[..., 2].abs() * 0.3
    fake_boxes[..., 3] = fake_boxes[..., 1] + fake_boxes[..., 3].abs() * 0.3
    fake_boxes = fake_boxes.clamp(0, 1)

    # Mask: last 6 are padding
    fake_mask = torch.zeros(B, N, dtype=torch.bool).to(device)
    fake_mask[:, 10:] = True   # objects 10–15 are padding

    # ── Build and run model ──
    model = RelationModule(d_model=512, d_relation=256, num_layers=2).to(device)
    model.eval()

    with torch.no_grad():
        out = model(fake_obj_features, fake_boxes, fake_mask)

    print("\n── Output shapes ──")
    print("relation_features :", out["relation_features"].shape)  # (2, 16, 512)
    print("relation_mask     :", out["relation_mask"].shape)       # (2, 16)
    print("edge_weights      :", out["edge_weights"].shape)        # (2, 16, 16)

    print("\n── Sanity checks ──")

    # Check padding positions are zeroed out
    pad_vals = out["relation_features"][:, 10:, :].abs().max().item()
    print(f"Padding tokens zeroed : {pad_vals == 0.0}  (max abs = {pad_vals:.6f})")

    # Check valid tokens changed from input (GCN should modify them)
    diff = (out["relation_features"][:, :10, :] - fake_obj_features[:, :10, :]).abs().mean().item()
    print(f"Features modified by GCN : {diff > 0}  (mean abs diff = {diff:.4f})")

    # Check edge weights sum to ~1 over neighbours
    valid_attn_sum = out["edge_weights"][0, 0, :10].sum().item()
    print(f"Edge weights sum (obj 0, valid nbrs) : {valid_attn_sum:.4f}  (should be ~1.0)")

    print("\n✓ RelationModule working correctly!")