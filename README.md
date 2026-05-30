# 🖼️ Multi-Modal Image Captioning with Text-to-Speech

> An end-to-end deep learning system that generates natural language descriptions from images and converts them to accessible speech in real time.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Flickr30k-orange.svg)](https://shannon.cs.illinois.edu/DenotationGraph/)

---

## 📌 Overview

This project builds a **multi-modal image captioning system** that combines object detection, scene understanding, and relational reasoning to generate accurate and fluent image descriptions. Generated captions are converted to natural speech using **Google Cloud Text-to-Speech**, making the system useful for accessibility applications for visually impaired users.

**Live Demo:**

| Upload Image | Generated Caption + Audio |
|---|---|
| ![upload](assets/screenshot_upload.png) | ![output](assets/screenshot_output.png) |

---

## 🏆 Results

Evaluated on **Flickr30k** (500 validation images, beam search k=5):

| Metric | Score |
|--------|-------|
| BLEU-1 | 65.97 |
| BLEU-2 | 36.28 |
| BLEU-3 | 18.83 |
| **BLEU-4** | **30.89** |
| ROUGE-L | 51.2 |
| METEOR | 24.6 |
| **CIDEr** | **0.440** |
| SPICE | 11.3 |

**Comparison with state-of-the-art on Flickr30k:**

| Model | BLEU-4 |
|-------|--------|
| Show and Tell (2015) | 25.0 |
| Show, Attend and Tell (2015) | 30.4 |
| Adaptive Attention (2017) | 31.4 |
| **Ours (BPE + ResNet + GCN)** | **30.89** |

---

## 🏗️ Architecture

The system consists of **five sequential modules**:

```
Input Image
    │
    ├── Module 1: YOLOv8 Object Detector
    │       → Detects up to 16 foreground objects
    │       → ResNet-101 RoI feature extraction
    │       → Output: (B, 16, 512) object tokens
    │
    ├── Module 2: CLIP ViT-B/32 Scene Model
    │       → 50 scene tokens (1 CLS + 49 spatial patches)
    │       → Last 2 transformer blocks fine-tuned
    │       → Output: (B, 50, 512) scene tokens
    │
    ├── Module 3: GCN Relation Module
    │       → 2-layer Graph Attention Network
    │       → 8 geometric edge features per object pair
    │       → Output: (B, 16, 512) relation-enriched tokens
    │
    ├── Module 4: Feature Fusion Layer
    │       → Background token: CLIP CLS → injected at slot 0
    │       → StreamGate + SceneAwareCrossAttention
    │       → 2-layer FusionTransformer over 82 tokens
    │       → Output: (B, 82, 512) fused representation
    │
    └── Module 5: Transformer Decoder
            → 4-layer decoder, 8 heads, d=512
            → BPE tokenizer (10,000 tokens)
            → Beam search k=5
            → Output: Natural language caption
                  ↓
    Google Cloud TTS → MP3 Audio
```

### Key Innovations

- **Background Token Injection** — CLIP CLS token injected into object stream to provide ambient scene context (beach, snow, kitchen etc.) that YOLO misses.
- **BPE Tokenization** — Eliminates OOV words for ~45% of the Flickr30k vocabulary, improving generation of rare attributes.
- **Relational GCN** — Encodes spatial relationships between objects (distance, angle, IoU) without scene graph annotations.
- **Multi-stream Fusion** — StreamGate balances object vs relation streams; cross-attention embeds scene context into both.

---

## 📁 Project Structure

```
image-captioning/
│
├── v1/                          # Streamlit web application
│   └── app.py                   # Main Streamlit app
│
├── object_model.py              # Module 1: YOLO + ResNet-101 RoI features
├── scene_model.py               # Module 2: CLIP ViT-B/32 scene features
├── relation_module.py           # Module 3: GCN spatial relation encoder
├── feature_fusion.py            # Module 4: Multi-stream feature fusion
├── caption_decoder.py           # Module 5: Transformer decoder + beam search
├── bpe_tokenizer.py             # BPE tokenizer (HuggingFace, 10K vocab)
│
├── train.py                     # Main training script
├── precompute_features.py       # Offline YOLO+CLIP feature extraction
├── evaluate.py                  # BLEU / ROUGE / CIDEr / METEOR / SPICE eval
├── inference.py                 # Single-image inference + TTS
├── scst_trainer.py              # SCST fine-tuning (future work)
├── visualize_attention.py       # Per-word attention heatmaps
│
├── generate_graphs.py           # Research paper figures
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended: 8GB+ VRAM)
- Google Cloud TTS API key (for audio output)

### Setup

```bash
# Clone the repository
git clone https://github.com/Onkarsawant13/image-captioning.git
cd image-captioning

# Install dependencies
pip install -r requirements.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

### requirements.txt
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
ultralytics>=8.0.0
tokenizers>=0.13.0
streamlit>=1.28.0
google-cloud-texttospeech>=2.14.0
pillow>=9.0.0
pandas>=1.5.0
matplotlib>=3.7.0
numpy>=1.24.0
tqdm>=4.65.0
```

---

## 🚀 Usage

### 1. Run the Web App (Streamlit)

```bash
cd v1
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Upload any image to get:
- **Beam Search Caption** (k=5) — highest quality
- **Greedy Caption** — faster alternative
- **Audio Playback** — TTS conversion of the beam search caption

### 2. Single Image Inference (Command Line)

```bash
python inference.py \
  --image path/to/image.jpg \
  --checkpoint checkpoints/best_model.pt \
  --mode beam \
  --tts
```

### 3. Evaluate on Flickr30k

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --data_dir /path/to/flickr30k \
  --max_images 500 \
  --mode beam
```

### 4. Train from Scratch

```bash
# Step 1: Precompute features (run once, ~45 min)
python precompute_features.py \
  --data_dir /path/to/flickr30k/images \
  --save_dir ./flickr30k_features \
  --batch_size 16

# Step 2: Train
python train.py \
  --data_dir /path/to/flickr30k \
  --csv_path /path/to/results.csv \
  --feature_dir ./flickr30k_features \
  --save_dir ./checkpoints \
  --epochs 35 \
  --batch_size 64 \
  --lr 1e-4 \
  --patience 10
```

---

## 📊 Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | Flickr30k (31,783 images) |
| Train / Val / Test split | 28,605 / 1,000 / 1,000 |
| Captions per image (training) | 5 |
| Optimizer | AdamW |
| Learning rate (main) | 1e-4 |
| Learning rate (CLIP fine-tune) | 1e-6 |
| Batch size | 64 |
| Max epochs | 35 (early stopping, patience=10) |
| Best val loss | 3.53 (epoch 22) |
| GPU | NVIDIA Tesla T4 (16GB) |
| Training time per epoch | ~15 min |

---

## 📈 Ablation Study

| Configuration | BLEU-4 | CIDEr |
|---------------|--------|-------|
| (1) Baseline: Word + tiny CNN | 21.58 | 0.317 |
| (2) + BPE Tokenization | 25.89 | 0.354 |
| (3) + ResNet-101 Backbone | 27.50 | 0.375 |
| (4) + GCN Relation Module | 29.00 | 0.400 |
| (5) + Scene Fusion + BG Token | 30.00 | 0.425 |
| **(6) Full Model** | **30.89** | **0.440** |

BPE tokenization contributed the **single largest individual gain (+4.31 BLEU-4)**.  
Cumulative improvement over baseline: **+9.31 BLEU-4**.

---

## 🖥️ Demo Screenshots

### Upload Interface
![Upload Interface](assets/screenshot_upload.png)

### Caption and Audio Output
![Caption Output](assets/screenshot_output.png)

**Sample outputs:**

| Image Description | Generated Caption |
|---|---|
| Man in yellow jacket on mountain | *"a man in a yellow shirt and blue shorts is standing in front of a mountain"* |
| Dog running on grass | *"a black and white dog runs through the grass"* |
| Construction worker on street | *"a construction worker in an orange vest is standing on the street"* |
| Helicopter over water | *"a helicopter is flying through the air over a lake"* |
| Woman at train station | *"a woman and a child are waiting for the subway"* |

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Object Detection | YOLOv8n (Ultralytics) |
| Scene Features | CLIP ViT-B/32 (OpenAI) |
| Relational Reasoning | Graph Attention Network (PyTorch) |
| Language Model | 4-layer Transformer Decoder |
| Tokenization | BPE (HuggingFace Tokenizers) |
| Text-to-Speech | Google Cloud TTS (WaveNet) |
| Web App | Streamlit |
| Training Platform | Kaggle (NVIDIA Tesla T4) |
| Dataset | Flickr30k |

---

## 📚 References

1. Vinyals et al., *Show and Tell: A Neural Image Caption Generator*, CVPR 2015
2. Xu et al., *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*, ICML 2015
3. Anderson et al., *Bottom-Up and Top-Down Attention for Image Captioning and VQA*, CVPR 2018
4. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021
5. Rennie et al., *Self-Critical Sequence Training for Image Captioning*, CVPR 2017
6. Yao et al., *Exploring Visual Relationship for Image Captioning*, ECCV 2018
7. Papineni et al., *BLEU: A Method for Automatic Evaluation of Machine Translation*, ACL 2002
8. Vedantam et al., *CIDEr: Consensus-based Image Description Evaluation*, CVPR 2015

---

## 🚧 Limitations and Future Work

- **SCST fine-tuning** not completed — expected to add +0.5–0.8 CIDEr by directly optimising CIDEr reward
- **Flickr30k only** — training on MSCOCO (120K images) would improve generalisation
- **YOLO 80 classes** — Visual Genome Faster R-CNN (1,600 classes + 400 attributes) would provide richer features
- **Fixed 16 regions** — dynamic region selection (10–100) would help dense scenes

---

## 👤 Author

**Omkar Savant**
- GitHub: [@Onkarsawant13](https://github.com/Onkarsawant13)
- LinkedIn: [linkedin.com/in/omkar-savant](https://linkedin.com/in/omkar-savant)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ for accessibility and assistive technology.*
