"""
Inference Script — Image Captioning (v1 — BLEU-4 30.89)
=========================================================
Compatible with checkpoints trained using:
  - Word-level tokenizer
  - Tiny CNN RoI encoder
  - v1 FeatureFusionLayer (187 param keys)
  - Simple TransformerDecoder

Usage:
    python inference.py --image photo.jpg \
                        --checkpoint ./checkpoints/best_model.pt

    python inference.py --image photo.jpg --show_objects
    python inference.py --image a.jpg b.jpg c.jpg --compare
    python inference.py --folder ./images/ --save results.txt
"""

import os
import sys
import json
import argparse
import textwrap
from pathlib import Path

import torch
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from object_model    import ObjectModel
from scene_model     import SceneModel
from relation_module import RelationModule
from feature_fusion  import FeatureFusionLayer
from caption_decoder import ImageCaptionModel


# ── Tokenizer (word-level) ────────────────────────────────────────
class Tokenizer:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.pad_id  = 0
        self.bos_id  = 1
        self.eos_id  = 2
        self.unk_id  = 3

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.word2id = data["word2id"]
        self.id2word = {int(k): v for k, v in data["id2word"].items()}
        self.actual_vocab_size = len(self.word2id)

    def decode(self, ids):
        words = []
        for i in ids:
            if i == self.eos_id:
                break
            if i not in (self.pad_id, self.bos_id):
                words.append(self.id2word.get(i, "<unk>"))
        return " ".join(words)


# ── Full model ────────────────────────────────────────────────────
class CaptioningModel(torch.nn.Module):
    def __init__(self, vocab_size, pad_id=0, bos_id=1, eos_id=2):
        super().__init__()
        self.object_model    = ObjectModel(
            yolo_weights="yolov8n.pt", d_model=512, max_objects=16)
        self.scene_model     = SceneModel(
            clip_model_name="ViT-B/32", d_model=512)
        self.relation_module = RelationModule(
            d_model=512, d_relation=256, num_layers=2)
        self.fusion_layer    = FeatureFusionLayer(
            d_model=512, num_heads=8, num_layers=2)
        self.caption_model   = ImageCaptionModel(
            vocab_size=vocab_size, d_model=512, num_layers=4,
            num_heads=8, max_seq_len=50,
            pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)

    @torch.no_grad()
    def caption(self, pil_images, device, beam_size=5):
        obj_out   = self.object_model(pil_images, device=device)
        scene_out = self.scene_model(pil_images, device=device)
        rel_out   = self.relation_module(
            obj_out["object_features"],
            obj_out["boxes"],
            obj_out["object_mask"])
        fused_out = self.fusion_layer(obj_out, scene_out, rel_out)
        if beam_size > 1:
            seqs = self.caption_model.generate_beam(fused_out, device, beam_size)
        else:
            raw  = self.caption_model.generate_greedy(fused_out, device)
            seqs = [raw[i].tolist() for i in range(raw.shape[0])]
        return seqs, obj_out


# ── Loader ────────────────────────────────────────────────────────
def load_model(checkpoint_path, device):
    ckpt_path = Path(checkpoint_path)
    candidates = [
        Path(str(ckpt_path).replace(".pt", "_tokenizer.json")),
        ckpt_path.parent / "best_model_tokenizer.json",
        ckpt_path.parent / "final_model_tokenizer.json",
    ] + list(ckpt_path.parent.glob("*tokenizer*.json"))
    tok_path = next((p for p in candidates if p.exists()), None)
    if tok_path is None:
        raise FileNotFoundError(f"Tokenizer not found near {ckpt_path}")

    tokenizer = Tokenizer()
    tokenizer.load(str(tok_path))
    print(f"  Tokenizer  : {tok_path.name}  ({tokenizer.actual_vocab_size:,} tokens)")

    ckpt  = torch.load(str(ckpt_path), map_location="cpu")
    epoch = ckpt.get("epoch", "?")
    vloss = ckpt.get("val_loss", "?")
    print(f"  Checkpoint : {ckpt_path.name}  (epoch {epoch+1 if isinstance(epoch,int) else epoch}, val_loss {vloss:.4f})")

    model = CaptioningModel(
        vocab_size=tokenizer.actual_vocab_size,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
    ).to(device)

    model_state = model.state_dict()
    matched = {k: v for k, v in ckpt["model_state"].items()
               if k in model_state and v.shape == model_state[k].shape}
    model_state.update(matched)
    model.load_state_dict(model_state)
    print(f"  Weights    : {len(matched)}/{len(model_state)} loaded ({100*len(matched)/len(model_state):.1f}%)")
    model.eval()
    return model, tokenizer


# ── YOLO class names ──────────────────────────────────────────────
YOLO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def get_objects(obj_out):
    class_ids = obj_out.get("class_ids", None)
    obj_mask  = obj_out.get("object_mask", None)
    if class_ids is None: return []
    seen, names = set(), []
    for i in range(class_ids.shape[1]):
        if obj_mask is not None and obj_mask[0, i].item(): continue
        cid = class_ids[0, i].item()
        if 0 <= cid < len(YOLO_CLASSES) and YOLO_CLASSES[cid] not in seen:
            names.append(YOLO_CLASSES[cid])
            seen.add(YOLO_CLASSES[cid])
    return names


# ── Display ───────────────────────────────────────────────────────
LINE  = "─" * 60
DLINE = "═" * 60

def print_banner(args, device):
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n{DLINE}")
    print(f"  IMAGE CAPTIONING — Inference")
    print(f"  Architecture : YOLOv8 + CLIP ViT-B/32 + GCN + Transformer")
    print(f"  Trained on   : Flickr30k  (31,000 images)")
    print(f"  BLEU-4       : 30.89  |  CIDEr: 0.44")
    print(f"  Device       : {gpu}")
    print(f"  Checkpoint   : {Path(args.checkpoint).name}")
    mode = "Beam Search (k=5)" if args.mode == "beam" else "Greedy"
    print(f"  Mode         : {mode}")
    print(f"{DLINE}\n")

def print_result(img_path, caption, objects=None, idx=None, total=None):
    counter = f"[{idx}/{total}] " if idx is not None else ""
    print(f"\n{LINE}")
    print(f"  {counter}{Path(img_path).name}")
    if objects:
        print(f"  Detected : {', '.join(objects[:8])}")
    wrapped = textwrap.fill(caption, width=54,
                            initial_indent="  Caption  : ",
                            subsequent_indent="             ")
    print(wrapped)
    print(LINE)


# ── Main inference ────────────────────────────────────────────────
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_banner(args, device)
    print("  Loading model...")
    model, tokenizer = load_model(args.checkpoint, device)
    print("  Model ready.\n")

    image_paths = []
    for p in (args.image or []):
        if Path(p).exists(): image_paths.append(p)
        else: print(f"  Warning: not found — {p}")
    if args.folder:
        exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        found = sorted(f for f in Path(args.folder).iterdir()
                       if f.suffix.lower() in exts)
        image_paths += [str(f) for f in found]
        print(f"  Found {len(found)} images in {Path(args.folder).name}/\n")

    if not image_paths:
        print("  No images found.")
        return

    total     = len(image_paths)
    results   = []
    beam_size = 5 if args.mode == "beam" else 1

    for idx, img_path in enumerate(image_paths, 1):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Could not open {img_path}: {e}")
            continue
        try:
            seqs, obj_out = model.caption([img], device, beam_size=beam_size)
            caption = tokenizer.decode(seqs[0])
            if not caption.strip():
                caption = "(model still learning)"
        except Exception as e:
            print(f"  Inference failed: {e}")
            continue

        objects = get_objects(obj_out) if args.show_objects else None
        print_result(img_path, caption, objects, idx, total)
        results.append((Path(img_path).name, caption, objects or []))
        if device.type == "cuda": torch.cuda.empty_cache()

    print(f"\n{DLINE}")
    print(f"  Done — captioned {len(results)}/{total} images")
    print(f"{DLINE}\n")

    if args.save and results:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write("Image Captioning Results\n" + "="*60 + "\n")
            f.write(f"BLEU-4: 30.89  CIDEr: 0.44\n" + "="*60 + "\n\n")
            for name, cap, objs in results:
                f.write(f"Image   : {name}\n")
                if objs: f.write(f"Objects : {', '.join(objs)}\n")
                f.write(f"Caption : {cap}\n")
                f.write("-"*60 + "\n")
        print(f"  Saved → {args.save}\n")


# ── Compare mode ──────────────────────────────────────────────────
def run_compare(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n{DLINE}")
    print(f"  COMPARE MODE — Greedy vs Beam Search")
    print(f"  Device: {gpu}")
    print(f"{DLINE}\n")
    print("  Loading model...")
    model, tokenizer = load_model(args.checkpoint, device)
    print("  Model ready.\n")

    for img_path in (args.image or []):
        if not Path(img_path).exists():
            print(f"  Not found: {img_path}")
            continue
        img  = Image.open(img_path).convert("RGB")
        name = Path(img_path).name
        seqs_g, obj_out = model.caption([img], device, beam_size=1)
        greedy_cap = tokenizer.decode(seqs_g[0]) or "(empty)"
        seqs_b, _  = model.caption([img], device, beam_size=5)
        beam_cap   = tokenizer.decode(seqs_b[0]) or "(empty)"
        objects    = get_objects(obj_out)
        print(f"\n{LINE}")
        print(f"  Image    : {name}")
        if objects: print(f"  Detected : {', '.join(objects[:8])}")
        print(f"  Greedy   : {greedy_cap}")
        print(f"  Beam (5) : {beam_cap}")
        print(LINE)
        if device.type == "cuda": torch.cuda.empty_cache()
    print()


# ── CLI ───────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Image Captioning Inference (v1 BLEU-4 30.89)")
    p.add_argument("--image",        nargs="+", default=None)
    p.add_argument("--folder",       type=str,  default=None)
    p.add_argument("--checkpoint",   type=str,
                   default="./checkpoints/best_model.pt")
    p.add_argument("--mode",         type=str,  default="beam",
                   choices=["beam", "greedy"])
    p.add_argument("--show_objects", action="store_true")
    p.add_argument("--save",         type=str,  default=None)
    p.add_argument("--compare",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    if not args.image and not args.folder:
        print("\n  Usage: python inference.py --image photo.jpg\n")
        import sys; sys.exit(1)
    if args.compare:
        run_compare(args)
    else:
        run_inference(args)