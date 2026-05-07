"""
Image Captioning for Visually Impaired — Streamlit Demo App
============================================================
Run: streamlit run app.py

Place this file in the same folder as your model .py files and checkpoints.
Generates captions using both Greedy and Beam Search decoding,
then speaks the Beam Search caption aloud via Google Text-to-Speech (gTTS).
"""

import sys
import io
import json
import time
import base64
from pathlib import Path

import torch
from PIL import Image
import streamlit as st
from gtts import gTTS

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Captioning for Visually Impaired",
    page_icon="🔊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main {
    background: #0f0f0f;
}

.stApp {
    background: #0f0f0f;
}

/* Header */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #f5f0e8;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #6b6b6b;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Upload zone */
.upload-zone {
    border: 1.5px dashed #2a2a2a;
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    background: #141414;
    transition: border-color 0.3s;
}

/* Caption card */
.caption-card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-top: 1.2rem;
    position: relative;
}

.caption-label {
    font-size: 0.7rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 500;
    margin-bottom: 0.6rem;
}

.caption-text {
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem;
    color: #f5f0e8;
    line-height: 1.5;
    font-style: italic;
}

.caption-text-greedy {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: #999;
    line-height: 1.6;
    font-weight: 300;
}

/* Audio section */
.audio-section {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-top: 1.4rem;
    position: relative;
    overflow: hidden;
}

.audio-section::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(200,184,138,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.audio-label {
    font-size: 0.7rem;
    color: #7a7a9a;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 500;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.audio-label .speaker-icon {
    font-size: 1rem;
    animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
    0%, 100% { opacity: 0.7; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.15); }
}

.audio-caption-preview {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #c8b88a;
    line-height: 1.5;
    font-style: italic;
    margin-bottom: 1rem;
    position: relative;
    z-index: 1;
}

/* Style the audio element */
audio {
    width: 100%;
    border-radius: 8px;
    outline: none;
    height: 40px;
}

/* Metric pills */
.metric-row {
    display: flex;
    gap: 0.8rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}

.metric-pill {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 100px;
    padding: 0.35rem 1rem;
    font-size: 0.78rem;
    color: #888;
    font-weight: 400;
}

.metric-pill span {
    color: #c8b88a;
    font-weight: 500;
}

/* Architecture badge */
.arch-badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 0.25rem 0.7rem;
    font-size: 0.72rem;
    color: #555;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    font-weight: 400;
    letter-spacing: 0.03em;
}

/* Divider */
.thin-divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 2rem 0;
}

/* Hide streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stFileUploader label { color: #666 !important; }
.stSpinner > div { border-top-color: #c8b88a !important; }

/* Image rounded */
.stImage img {
    border-radius: 10px;
}

/* Button */
.stButton > button {
    background: #c8b88a !important;
    color: #0f0f0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.8rem !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #141414 !important;
    border-color: #2a2a2a !important;
    color: #f5f0e8 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent))

@st.cache_resource(show_spinner=False)
def load_model_cached(checkpoint_path):
    """Load model once and cache it."""
    from object_model    import ObjectModel
    from scene_model     import SceneModel
    from relation_module import RelationModule
    from feature_fusion  import FeatureFusionLayer
    from caption_decoder import ImageCaptionModel

    # Load tokenizer
    ckpt_path = Path(checkpoint_path)
    candidates = [
        Path(str(ckpt_path).replace(".pt", "_tokenizer.json")),
        ckpt_path.parent / "best_model_tokenizer.json",
    ] + list(ckpt_path.parent.glob("*tokenizer*.json"))
    tok_path = next((p for p in candidates if p.exists()), None)
    if tok_path is None:
        raise FileNotFoundError(f"Tokenizer not found near {ckpt_path}")

    with open(tok_path) as f:
        tok_data = json.load(f)
    word2id = tok_data["word2id"]
    id2word = {int(k): v for k, v in tok_data["id2word"].items()}
    vocab_size = len(word2id)
    pad_id, bos_id, eos_id = 0, 1, 2

    def decode(ids):
        words = []
        for i in ids:
            if i == eos_id: break
            if i not in (pad_id, bos_id):
                words.append(id2word.get(i, "<unk>"))
        return " ".join(words)

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class FullModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.object_model    = ObjectModel("yolov8n.pt", d_model=512, max_objects=16)
            self.scene_model     = SceneModel("ViT-B/32", d_model=512)
            self.relation_module = RelationModule(d_model=512, d_relation=256, num_layers=2)
            self.fusion_layer    = FeatureFusionLayer(d_model=512, num_heads=8, num_layers=2)
            self.caption_model   = ImageCaptionModel(
                vocab_size=vocab_size, d_model=512, num_layers=4,
                num_heads=8, max_seq_len=50,
                pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)

        @torch.no_grad()
        def generate(self, pil_images, beam_size=5):
            obj_out   = self.object_model(pil_images, device=device)
            scene_out = self.scene_model(pil_images, device=device)
            rel_out   = self.relation_module(
                obj_out["object_features"],
                obj_out["boxes"], obj_out["object_mask"])
            fused_out = self.fusion_layer(obj_out, scene_out, rel_out)
            if beam_size > 1:
                return self.caption_model.generate_beam(fused_out, device, beam_size)
            else:
                raw = self.caption_model.generate_greedy(fused_out, device)
                return [raw[i].tolist() for i in range(raw.shape[0])]

    model = FullModel().to(device)
    ckpt  = torch.load(str(ckpt_path), map_location="cpu")
    model_state = model.state_dict()
    matched = {k: v for k, v in ckpt["model_state"].items()
               if k in model_state and v.shape == model_state[k].shape}
    model_state.update(matched)
    model.load_state_dict(model_state)
    model.eval()

    epoch    = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", 0)

    return model, decode, device, epoch, val_loss


# ── Header ────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Image Captioning 🔊</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Assistive Vision · Text-to-Speech</div>', unsafe_allow_html=True)

st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

# ── Sidebar — settings ───────────────────────────────────────────
checkpoint_dir = Path(r"D:\image_captioning _final\v1")
checkpoint_path = str(checkpoint_dir / "epoch_15.pt")

with st.sidebar:
    st.markdown("### 🔊 Settings")
    tts_speed = st.checkbox("Slow speech", value=False, help="Speak the caption more slowly")


# ── Load model ────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    try:
        model, decode, device, _, _ = load_model_cached(checkpoint_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


# ── Upload ────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(pil_image, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div class="caption-label">File</div>
            <div style="color: #888; font-size: 0.9rem;">{uploaded.name}</div>
        </div>
        <div style="margin-bottom: 1rem;">
            <div class="caption-label">Size</div>
            <div style="color: #888; font-size: 0.9rem;">{pil_image.width} × {pil_image.height}px</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    # Generate BOTH captions — always
    with st.spinner("Generating captions..."):
        t0 = time.time()
        try:
            # Beam search caption (used for TTS)
            seqs_beam = model.generate([pil_image], beam_size=5)
            beam_caption = decode(seqs_beam[0])

            # Greedy caption
            seqs_greedy = model.generate([pil_image], beam_size=1)
            greedy_caption = decode(seqs_greedy[0])

            elapsed = time.time() - t0

            if not beam_caption.strip():
                beam_caption = "model is still learning — try more training epochs"
            if not greedy_caption.strip():
                greedy_caption = "model is still learning — try more training epochs"

        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    # ── Generate TTS audio from beam search caption ──────────────
    tts_audio_bytes = None
    try:
        tts = gTTS(text=beam_caption, lang='en', slow=tts_speed)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        tts_audio_bytes = audio_buffer.read()
    except Exception as e:
        st.warning(f"Text-to-Speech failed: {e}")

    # ── Display results ──────────────────────────────────────────
    # Beam Search Caption (primary — spoken aloud)
    st.markdown(f"""
    <div class="caption-card">
        <div class="caption-label">🔍 Beam Search Caption (k=5)</div>
        <div class="caption-text">"{beam_caption}"</div>
    </div>
    """, unsafe_allow_html=True)

    # Greedy Caption (secondary)
    st.markdown(f"""
    <div class="caption-card" style="margin-top: 0.8rem;">
        <div class="caption-label">⚡ Greedy Caption</div>
        <div class="caption-text-greedy">"{greedy_caption}"</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Audio player section ─────────────────────────────────────
    if tts_audio_bytes:
        st.markdown(f"""
        <div class="audio-section">
            <div class="audio-label">
                <span class="speaker-icon">🔊</span>
                LISTENING — BEAM SEARCH CAPTION
            </div>
            <div class="audio-caption-preview">"{beam_caption}"</div>
        </div>
        """, unsafe_allow_html=True)
        st.audio(tts_audio_bytes, format="audio/mp3", autoplay=True)

    st.markdown("<br>", unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">🔊</div>
        <div style="color: #555; font-size: 0.9rem; font-weight: 300;">
            Drop an image here or click Browse files above
        </div>
        <div style="color: #3a3a3a; font-size: 0.78rem; margin-top: 0.5rem;">
            JPG · PNG · BMP · WEBP
        </div>
        <div style="color: #c8b88a; font-size: 0.82rem; margin-top: 1rem; font-weight: 400;">
            🔊 The caption will be spoken aloud automatically
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Show example info
    st.markdown("""
    <div style="color: #3a3a3a; font-size: 0.82rem; line-height: 1.8;">
        Upload any photo and the model will generate a natural language description.<br>
        The architecture combines object detection, scene understanding,
        spatial relationships, and a transformer decoder trained on 31,000 Flickr images.<br><br>
        <strong style="color: #555;">🔊 Accessibility:</strong> 
        <span style="color: #555;">Both Greedy and Beam Search captions are generated.
        The higher-quality Beam Search caption is automatically converted to speech.</span>
    </div>
    """, unsafe_allow_html=True)
