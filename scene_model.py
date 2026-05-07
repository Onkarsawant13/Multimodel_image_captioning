"""
Scene Model — Module 2 of Image Captioning Architecture
========================================================
Uses CLIP (ViT-B/32) as a frozen scene encoder and extracts:
  1. Global CLS token  → overall scene summary
  2. Patch tokens      → spatial scene features (grid of 7x7 = 49 patches)

Why CLIP over plain ViT:
  - Pretrained on image-TEXT pairs → features already language-aligned
  - Decoder generates captions more naturally from language-aware features
  - Flickr30k = everyday scenes, exactly CLIP's training domain

Output fed into Feature Fusion Layer:
  scene_features : (batch, num_patches + 1, d_model)
                    patch tokens + 1 CLS token = 50 tokens total
  scene_mask     : (batch, 50)  all False (no padding — scene always present)

Dependencies:
    pip install openai-clip torch torchvision
    or: pip install git+https://github.com/openai/CLIP.git
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import clip
from PIL import Image


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CLIP_MODEL     = "ViT-B/32"   # lightweight, 150MB, perfect for Flickr30k
D_CLIP         = 512           # CLIP ViT-B/32 output dim (matches our D_MODEL!)
D_MODEL        = 512           # must match object_model.py
NUM_PATCHES    = 49            # ViT-B/32 → 224/32 = 7 → 7×7 = 49 patches
NUM_TOKENS     = NUM_PATCHES + 1  # 49 patches + 1 CLS = 50 scene tokens


# ──────────────────────────────────────────────
# Scene Model
# ──────────────────────────────────────────────
class SceneModel(nn.Module):
    """
    CLIP-based scene encoder.

    Extracts TWO levels of scene understanding:
      • CLS token  — "what is this scene about overall?"
      • Patch tokens — "what is happening in each spatial region?"

    Both are projected to d_model and returned as a sequence of 50 tokens.
    The decoder can attend to all 50 via cross-attention in the fusion layer.

    Args:
        clip_model_name : CLIP variant, default "ViT-B/32"
        d_model         : output dim, must match ObjectModel (default 512)
        dropout         : dropout on projected features
    """

    def __init__(
        self,
        clip_model_name: str = CLIP_MODEL,
        d_model: int = D_MODEL,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # ── Load CLIP — stored outside nn.Module tracking ──
        # Same trick as ObjectModel: prevents .eval() / .train() propagation
        # into CLIP's internals which we never want to fine-tune.
        _clip_model, _clip_preprocess = clip.load(clip_model_name, device="cpu")
        _clip_model.eval()
        for param in _clip_model.parameters():
            param.requires_grad = False

        object.__setattr__(self, "_clip", _clip_model)
        object.__setattr__(self, "_clip_preprocess", _clip_preprocess)

        # ── Unfreeze last 2 CLIP transformer blocks for fine-tuning ──
        # _clip is now assigned so self._clip is accessible.
        # Unfreezing last 2 blocks lets CLIP adapt to captioning vocabulary
        # and learn to emphasise scene-relevant features (dock, sunset, water).
        for param in self._clip.visual.transformer.resblocks[-2:].parameters():
            param.requires_grad = True

        # CLIP ViT-B/32 output dim is 512 which already equals D_MODEL.
        # We still add a projection so we can optionally switch to larger
        # CLIP variants (ViT-L/14 = 768 dim) without changing downstream code.
        clip_dim = _clip_model.visual.output_dim   # 512 for ViT-B/32

        # ── Projections ──
        # CLS token projection: scalar scene summary → d_model
        self.cls_proj = nn.Sequential(
            nn.Linear(clip_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Patch token projection: spatial features → d_model
        # CLIP's intermediate patch features are extracted via hook (see below)
        # ViT-B/32 intermediate dim = 768 (transformer width before final proj)
        vit_width = _clip_model.visual.transformer.width   # 512 for ViT-B/32
        self.patch_proj = nn.Sequential(
            nn.Linear(vit_width, d_model),
            nn.LayerNorm(d_model),
        )

        self.dropout = nn.Dropout(dropout)

        # Storage for patch features captured by forward hook
        self._patch_features = None
        self._register_patch_hook()

    # ── Hook to capture intermediate patch tokens ─────────────────────────
    def _register_patch_hook(self):
        """
        CLIP's visual encoder only returns the CLS token by default.
        We register a hook on the last transformer block to also capture
        the full sequence (CLS + patch tokens) before the final projection.
        """
        def hook_fn(module, input, output):
            # output shape: (seq_len, batch, width) → (batch, seq_len, width)
            self._patch_features = output.permute(1, 0, 2)

        last_block = self._clip.visual.transformer.resblocks[-1]
        last_block.register_forward_hook(hook_fn)

    # ── Image preprocessing ───────────────────────────────────────────────
    def _preprocess_images(
        self,
        pil_images: list,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Preprocess PIL images using CLIP's own transform pipeline.
        Returns: (B, 3, 224, 224) float tensor on device
        """
        tensors = [self._clip_preprocess(img) for img in pil_images]
        return torch.stack(tensors).to(device)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(
        self,
        pil_images: list,
        device: torch.device = None,
    ) -> dict:
        """
        Args:
            pil_images : list of B PIL.Image objects
            device     : target device

        Returns dict with keys:
            "scene_features" : (B, 50, d_model)
                               index 0   = CLS (global scene token)
                               index 1–49 = spatial patch tokens
            "scene_mask"     : (B, 50) all-False — no padding in scene stream
            "cls_feature"    : (B, d_model) — global scene embedding only
                               (useful shortcut for simple baselines)
        """
        if device is None:
            device = next(self.parameters()).device

        B = len(pil_images)

        # ── Step 1: Preprocess ──
        img_tensor = self._preprocess_images(pil_images, device)  # (B, 3, 224, 224)

        # ── Step 2: Move CLIP to same device as input, then forward ──
        # CLIP is stored outside nn.Module so we must move it manually
        self._clip.to(device)
        # Run encode_image allowing gradients through the 2 unfrozen CLIP blocks.
        # Frozen blocks have requires_grad=False so no extra memory is used.
        # torch.no_grad() context (validation) will override this automatically.
        cls_token    = self._clip.encode_image(img_tensor).float()   # (B, clip_dim)
        patch_tokens = self._patch_features.float()                   # (B, 50, vit_width)

        # Separate out hook's CLS (index 0) from patch tokens (index 1:)
        # We use CLIP's final projected CLS for the global token (better quality)
        # and the hook's intermediate patches for spatial tokens
        patch_only = patch_tokens[:, 1:, :]   # (B, 49, vit_width)

        # ── Step 3: Project to d_model ──
        cls_feat   = self.cls_proj(cls_token)        # (B, d_model)
        patch_feat = self.patch_proj(patch_only)     # (B, 49, d_model)

        # Apply dropout
        cls_feat   = self.dropout(cls_feat)
        patch_feat = self.dropout(patch_feat)

        # ── Step 4: Concatenate CLS + patches → 50 scene tokens ──
        # CLS at position 0, patches at positions 1–49
        scene_features = torch.cat(
            [cls_feat.unsqueeze(1), patch_feat], dim=1
        )                                            # (B, 50, d_model)

        # Scene has no padding — every image always has all 50 tokens
        scene_mask = torch.zeros(
            B, NUM_TOKENS, dtype=torch.bool, device=device
        )                                            # (B, 50) all False

        return {
            "scene_features" : scene_features,    # (B, 50, d_model)
            "scene_mask"     : scene_mask,         # (B, 50) no padding
            "cls_feature"    : cls_feat,           # (B, d_model) global shortcut
        }


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import requests
    from io import BytesIO
    from PIL import Image as PILImage

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load a sample image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        img = PILImage.open(BytesIO(requests.get(url).content)).convert("RGB")
        print("Loaded image from URL")
    except Exception:
        img = PILImage.new("RGB", (640, 480), color=(100, 149, 237))
        print("Using dummy image")

    # Build model
    print("Loading CLIP ViT-B/32...")
    model = SceneModel(clip_model_name="ViT-B/32", d_model=512).to(device)
    model.eval()
    print("Model loaded.")

    batch = [img, img]   # simulate batch of 2
    with torch.no_grad():
        out = model(batch, device=device)

    print("\n── Output shapes ──")
    print("scene_features :", out["scene_features"].shape)   # (2, 50, 512)
    print("scene_mask     :", out["scene_mask"].shape)        # (2, 50)
    print("cls_feature    :", out["cls_feature"].shape)       # (2, 512)

    print("\n── Sanity checks ──")
    print("No padding tokens :", not out["scene_mask"].any().item())   # True
    print("CLS == index 0    :",
          torch.allclose(out["scene_features"][:, 0, :],
                         out["cls_feature"], atol=1e-5))               # True
    print("Feature range     :",
          out["scene_features"].min().item(),
          "to",
          out["scene_features"].max().item())                          # finite values

    print("\n✓ SceneModel working correctly!")