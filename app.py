import os
import gc
import random
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from collections import OrderedDict
from huggingface_hub import hf_hub_download
import gradio as gr

# -------------------------------------------------------
# 1. DEVICE & CLASS NAMES
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
NUM_CLASSES = len(CLASS_NAMES)


def pretty(raw):
    parts = raw.split("___")
    plant = parts[0].replace("_", " ").replace("(including sour)", "").strip()
    cond = parts[1].replace("_", " ") if len(parts) > 1 else raw
    return plant, cond


# -------------------------------------------------------
# 2. MODEL CONFIG — files pulled from HF Model Hub
# -------------------------------------------------------
HF_REPO = "deepak0027/plant-disease-models"

MODEL_CONFIG = {
    "🏆 ConvNeXt-Tiny (Champion)": {"file": "convnext_tiny_plantvillage.pth", "size": 224, "acc": "99.13%"},
    "🏛️ ResNet-50 (Robust Baseline)": {"file": "resnet50_plantvillage_final.pth", "size": 224, "acc": "97.01%"},
    "🧬 DenseNet-121 (Feature Reuse)": {"file": "densenet121_plantvillage_final.pth", "size": 224, "acc": "96.74%"},
    "🔬 Inception-V3 (High-Res 299x299)": {"file": "inceptionv3_plantvillage.pth", "size": 299, "acc": "93.60%"},
    "📱 MobileNet-V3 (Lightweight)": {"file": "mobilenetv3_plantvillage.pth", "size": 224, "acc": "96.69%"},
    "📦 VGG-16 (Classic Deep CNN)": {"file": "vgg16_plantvillage.pth", "size": 224, "acc": "97.57%"},
    "👁️ Swin Transformer (Vision ViT)": {"file": "swin_final_plantvillage_10epochs.pth", "size": 224, "acc": "98.10%"},
    "🧪 EfficientNet-B0 (Baseline CNN)": {"file": "cnn_baseline_plantvillage_10epochs.pth", "size": 224, "acc": "96.54%"},
}

LEADERBOARD = [
    {"rank": "🥇", "name": "ConvNeXt-Tiny", "type": "Modern CNN", "acc": 99.13, "loss": 0.0236},
    {"rank": "🥈", "name": "Swin Transformer", "type": "Vision Transformer", "acc": 98.10, "loss": 0.0839},
    {"rank": "🥉", "name": "VGG-16", "type": "Classic Deep CNN", "acc": 97.57, "loss": 0.0323},
    {"rank": "4", "name": "ResNet-50", "type": "CNN", "acc": 97.01, "loss": 0.0957},
    {"rank": "5", "name": "DenseNet-121", "type": "CNN", "acc": 96.74, "loss": 0.2030},
    {"rank": "6", "name": "MobileNet-V3", "type": "Lightweight CNN", "acc": 96.69, "loss": 0.0808},
    {"rank": "7", "name": "EfficientNet-B0", "type": "CNN (Baseline)", "acc": 96.54, "loss": 0.1466},
    {"rank": "8", "name": "Inception-V3", "type": "CNN", "acc": 93.60, "loss": 0.5975},
]

SPECIES = sorted({pretty(c)[0] for c in CLASS_NAMES})
CLASS_LABELS = sorted({f"{pretty(c)[0]} · {pretty(c)[1]}" for c in CLASS_NAMES})
CLASS_LABEL_TO_RAW = {f"{pretty(c)[0]} · {pretty(c)[1]}": c for c in CLASS_NAMES}

# -------------------------------------------------------
# 3. MODEL ARCHITECTURE BUILDER
# -------------------------------------------------------
def build_model_architecture(choice):
    if "ConvNeXt" in choice:
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
    elif "Inception" in choice:
        m = models.inception_v3(weights=None, init_weights=False)
        m.aux_logits = False
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, NUM_CLASSES))
    elif "ResNet" in choice:
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, NUM_CLASSES))
    elif "DenseNet" in choice:
        m = models.densenet121(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier.in_features, NUM_CLASSES))
    elif "MobileNet" in choice:
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    elif "VGG" in choice:
        m = models.vgg16(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, NUM_CLASSES)
    elif "Swin" in choice:
        m = models.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, NUM_CLASSES)
    elif "EfficientNet" in choice:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {choice}")
    return m


# -------------------------------------------------------
# 4. LOAD MODEL — downloads from HF Hub automatically
# -------------------------------------------------------
_MODEL_CACHE = {}

def load_model(choice):
    if choice in _MODEL_CACHE:
        return _MODEL_CACHE[choice]

    cfg = MODEL_CONFIG[choice]
    img_size = cfg["size"]
    filename = cfg["file"]

    print(f"📥 Downloading {filename} from HF Hub...")
    local_path = hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type="model")
    print(f"✅ Loaded from: {local_path}")

    model = build_model_architecture(choice)
    state_dict = torch.load(local_path, map_location=device)

    clean_sd = OrderedDict()
    for k, v in state_dict.items():
        clean_sd[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(clean_sd)
    model = model.to(device)
    model.eval()

    _MODEL_CACHE[choice] = (model, img_size)
    return model, img_size


def _predict_raw(image: Image.Image, choice: str, k=3):
    """Core tensor -> prediction helper reused by every feature."""
    model, img_size = load_model(choice)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
        topk_probs, topk_idx = torch.topk(probs, k=min(k, NUM_CLASSES), dim=1)
    return pred.item(), conf.item(), topk_probs, topk_idx


# -------------------------------------------------------
# 5. INFERENCE / RESULT CARD
# -------------------------------------------------------
def analyze_plant(image: Image.Image, selected_engine: str):
    if image is None:
        return _idle_card("Awaiting specimen — upload a leaf image to begin")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        pred_idx, confidence_raw, top_probs, top_idx = _predict_raw(image, selected_engine, k=3)
    except Exception as e:
        return _error_card(str(e))

    engine_label = selected_engine.split(" ", 1)[1] if " " in selected_engine else selected_engine

    raw = CLASS_NAMES[pred_idx]
    plant, condition = pretty(raw)
    confidence = confidence_raw * 100

    healthy = "healthy" in condition.lower()
    accent = "#39ff9d" if healthy else "#ff5470"
    accent_soft = "rgba(57,255,157,.14)" if healthy else "rgba(255,84,112,.14)"
    status = "SPECIMEN HEALTHY" if healthy else "PATHOGEN DETECTED"
    status_icon = "✓" if healthy else "⚠"
    action = (
        "No intervention required. Current care routine is optimal — keep monitoring weekly."
        if healthy
        else f"Isolate affected foliage immediately and begin treatment protocol for <b>{condition}</b>."
    )

    rows = ""
    for i in range(top_probs.shape[1]):
        idx = top_idx[0][i].item()
        p = top_probs[0][i].item() * 100
        plant_i, cond_i = pretty(CLASS_NAMES[idx])
        label_i = f"{plant_i} · {cond_i}"
        is_top = (i == 0)
        rows += f"""
        <div class="rank-row {'rank-row-top' if is_top else ''}" style="animation-delay:{0.55 + i*0.09}s;">
            <span class="rank-idx">{i+1:02d}</span>
            <span class="rank-label">{label_i}</span>
            <div class="rank-bar-track">
                <div class="rank-bar-fill" style="--target:{p:.1f}%; background:{accent if is_top else '#4b5a6a'};"></div>
            </div>
            <span class="rank-pct" style="color:{accent if is_top else '#8a96a3'};">{p:.1f}%</span>
        </div>"""

    copy_text = f"{plant} — {condition} ({confidence:.1f}% confidence, {engine_label})".replace("'", "\\'")

    return f"""
    <div class="result-wrap">
        <div class="scan-line"></div>
        <div class="result-header" style="--accent:{accent}; --accent-soft:{accent_soft};">
            <div class="status-pill">
                <span class="status-dot"></span>
                <span>{status_icon} {status}</span>
            </div>
            <span class="engine-chip">⚙ {engine_label}</span>
        </div>

        <div class="id-block" style="animation-delay:.08s;">
            <p class="eyebrow">Identified crop</p>
            <h1 class="crop-name">{plant}</h1>
        </div>

        <div class="id-block" style="animation-delay:.18s;">
            <p class="eyebrow">Diagnosis</p>
            <h2 class="condition-name" style="color:{accent};">{condition}</h2>
        </div>

        <div class="confidence-block" style="animation-delay:.28s;">
            <div class="confidence-top">
                <span class="conf-label">Model confidence</span>
                <span class="conf-value" style="color:{accent};">{confidence:.2f}%</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="--target:{confidence}%; background:linear-gradient(90deg, {accent}88, {accent});"></div>
            </div>
        </div>

        <div class="action-block" style="animation-delay:.4s; border-color:{accent}; background:{accent_soft};">
            <strong>💡 Recommended action</strong>
            <p>{action}</p>
        </div>

        <div class="rank-block" style="animation-delay:.5s;">
            <p class="eyebrow" style="margin-bottom:10px;">Top predictions</p>
            {rows}
        </div>

        <button class="copy-btn" style="animation-delay:.6s;" onclick="navigator.clipboard.writeText('{copy_text}'); this.classList.add('copied'); this.querySelector('span').innerText='Copied to clipboard'; setTimeout(()=>{{this.classList.remove('copied'); this.querySelector('span').innerText='Copy result';}},1600);">
            <span>Copy result</span>
        </button>
    </div>
    """


def _idle_card(msg="Awaiting specimen..."):
    return f"""
    <div class="idle-card">
        <div class="idle-orbit">
            <div class="idle-ring ring-1"></div>
            <div class="idle-ring ring-2"></div>
            <div class="idle-ring ring-3"></div>
            <span class="idle-emoji">🔬</span>
        </div>
        <h2>{msg}</h2>
        <p class="idle-sub">Drop a leaf photo on the left to run diagnostics</p>
    </div>"""


def _error_card(msg):
    return f"""
    <div class="idle-card error-card">
        <span class="idle-emoji">⚠️</span>
        <h2>Model failed to load</h2>
        <p class="idle-sub">{msg}</p>
    </div>"""


def _loading_card(label="Running inference"):
    return f"""
    <div class="skeleton-card">
        <div class="skel-top">
            <div class="skel-spinner"></div>
            <span class="skel-live-label">{label}<span class="skel-dots"><span>.</span><span>.</span><span>.</span></span></span>
        </div>
        <div class="skel-line skel-title"></div>
        <div class="skel-line" style="width:35%"></div>
        <div class="skel-block"></div>
        <div class="skel-line" style="width:70%"></div>
        <div class="skel-line" style="width:55%"></div>
        <div class="skel-line" style="width:40%"></div>
    </div>
    """


# -------------------------------------------------------
# 6. NEW FEATURE — COMPARE ALL ENGINES (ensemble consensus)
# -------------------------------------------------------
def compare_all_engines(image, progress=gr.Progress()):
    if image is None:
        return _idle_card("Upload a specimen first, then compare every engine at once")

    choices = list(MODEL_CONFIG.keys())
    predictions = []

    for i, choice in enumerate(choices):
        progress((i) / len(choices), desc=f"Running {choice.split(' ', 1)[-1]}…")
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pred_idx, conf, _, _ = _predict_raw(image, choice, k=1)
            raw = CLASS_NAMES[pred_idx]
            plant, condition = pretty(raw)
            predictions.append({
                "engine": choice.split(" ", 1)[-1],
                "acc": MODEL_CONFIG[choice]["acc"],
                "label": f"{plant} · {condition}",
                "healthy": "healthy" in condition.lower(),
                "conf": conf * 100,
                "error": None,
            })
        except Exception as e:
            predictions.append({"engine": choice.split(" ", 1)[-1], "error": str(e)})

    progress(1.0, desc="Done")

    valid = [p for p in predictions if not p.get("error")]
    labels = [p["label"] for p in valid]
    consensus_label = max(set(labels), key=labels.count) if labels else None
    agree_count = labels.count(consensus_label) if consensus_label else 0
    total = len(valid)

    if consensus_label:
        gr.Info(f"⚔️ Ensemble done — {agree_count}/{total} engines agree on {consensus_label}.")
    else:
        gr.Warning("⚔️ Ensemble comparison finished but no engine returned a usable prediction.")

    consensus_html = ""
    if consensus_label:
        plant_c, cond_c = consensus_label.split(" · ", 1)
        healthy_c = "healthy" in cond_c.lower()
        c_accent = "#39ff9d" if healthy_c else "#ff5470"
        pct = (agree_count / total * 100) if total else 0
        consensus_html = f"""
        <div class="consensus-banner" style="--c-accent:{c_accent};">
            <div class="consensus-left">
                <p class="eyebrow">Ensemble consensus · {agree_count}/{total} engines agree</p>
                <h2 style="color:{c_accent};">{consensus_label}</h2>
            </div>
            <div class="consensus-ring" style="--pct:{pct:.0f};">
                <span>{pct:.0f}%</span>
            </div>
        </div>
        """

    rows = ""
    for i, p in enumerate(predictions):
        if p.get("error"):
            rows += f"""
            <div class="engine-row engine-row-error" style="animation-delay:{i*0.06:.2f}s;">
                <span class="engine-row-name">{p['engine']}</span>
                <span class="engine-row-status">⚠ failed to load</span>
            </div>"""
            continue
        accent = "#39ff9d" if p["healthy"] else "#ff5470"
        agree = (p["label"] == consensus_label)
        rows += f"""
        <div class="engine-row {'engine-row-agree' if agree else ''}" style="animation-delay:{i*0.06:.2f}s;">
            <span class="engine-row-name">{p['engine']} <em>({p['acc']} val-acc)</em></span>
            <span class="engine-row-label" style="color:{accent};">{p['label']}</span>
            <div class="rank-bar-track engine-row-track">
                <div class="rank-bar-fill" style="--target:{p['conf']:.1f}%; background:{accent};"></div>
            </div>
            <span class="rank-pct" style="color:{accent};">{p['conf']:.1f}%</span>
            {'<span class="agree-tag">✓ agrees</span>' if agree else ''}
        </div>"""

    return f"""
    <div class="result-wrap">
        <div class="scan-line"></div>
        {consensus_html}
        <div class="rank-block" style="animation-delay:.2s;">
            <p class="eyebrow" style="margin-bottom:10px;">All 8 engines, ranked by load order</p>
            {rows}
        </div>
    </div>
    """


# -------------------------------------------------------
# 7. NEW FEATURE — ROBUSTNESS LAB (live Gaussian noise stress test)
# -------------------------------------------------------
def add_gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    noise = np.random.randn(*arr.shape).astype(np.float32) * sigma
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def run_robustness_test(image, selected_engine, sigma):
    if image is None:
        idle = _idle_card("Upload a specimen to begin the stress test")
        return idle, idle, None, "Upload an image, choose a noise level, then run the test."

    noisy_img = add_gaussian_noise(image, sigma)
    clean_html = analyze_plant(image, selected_engine)
    noisy_html = analyze_plant(noisy_img, selected_engine)

    verdict = (
        f"Gaussian noise σ={sigma:.2f} applied. Compare the two diagnostic cards — "
        f"a robust engine keeps the same diagnosis and a similar confidence level under noise."
    )
    gr.Info(f"🌪️ Stress test complete at σ={sigma:.2f} — compare the two cards above.")
    return clean_html, noisy_html, noisy_img, verdict


# -------------------------------------------------------
# 7b. GRAD-CAM EXPLAINABILITY — "where is the model looking?"
# -------------------------------------------------------
# Pure torch/numpy/PIL implementation — no extra dependency (no pytorch-grad-cam
# needed) so this doesn't touch requirements.txt at all.
GRADCAM_UNSUPPORTED = "Swin"  # transformer — needs attention rollout, not grad-based CAM

def _gradcam_target_layer(model, choice):
    """The last spatially-informative conv feature map for each architecture —
    this is exactly the 'CNN Target Layer' the README calls out for Grad-CAM."""
    if "ConvNeXt" in choice:
        return model.features[-1]
    if "ResNet" in choice:
        return model.layer4[-1]
    if "DenseNet" in choice:
        return model.features
    if "MobileNet" in choice:
        return model.features[-1]
    if "VGG" in choice:
        return model.features[-1]
    if "Inception" in choice:
        return model.Mixed_7c
    if "EfficientNet" in choice:
        return model.features[-1]
    return None


def _heat_colormap(v: np.ndarray) -> np.ndarray:
    """v in [0,1] HxW -> HxWx3 uint8 'hot' colormap (black->red->yellow->white),
    written by hand so no matplotlib/opencv dependency is needed."""
    r = np.clip(v * 3.0, 0, 1)
    g = np.clip(v * 3.0 - 1.0, 0, 1)
    b = np.clip(v * 3.0 - 2.0, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def compute_gradcam(image: Image.Image, selected_engine: str):
    """Returns (overlay_image, plant, condition, confidence_pct, note_or_None)."""
    if GRADCAM_UNSUPPORTED in selected_engine:
        return None, None, None, None, (
            "Grad-CAM needs a spatial conv feature map to work from, so it isn't available for the "
            "Swin Transformer in this viewer — transformer attention needs a different technique "
            "(attention rollout) rather than gradient-based CAM. Pick any of the other 7 engines."
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, img_size = load_model(selected_engine)
    target_layer = _gradcam_target_layer(model, selected_engine)
    if target_layer is None:
        return None, None, None, None, "No Grad-CAM target layer is defined for this engine yet."

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    rgb_image = image.convert("RGB")
    tensor = transform(rgb_image).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    activations, gradients = {}, {}

    def fwd_hook(_module, _inp, out):
        activations["v"] = out

    def bwd_hook(_module, _grad_in, grad_out):
        gradients["v"] = grad_out[0].detach()

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    if hasattr(target_layer, "register_full_backward_hook"):
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)
    else:
        h_bwd = target_layer.register_backward_hook(bwd_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, pred_idx].item()) * 100
        score = logits[0, pred_idx]
        score.backward()

        act = activations["v"][0].detach()          # C,H,W
        grad = gradients["v"][0].detach()            # C,H,W
        weights = grad.mean(dim=(1, 2))               # C
        cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
    finally:
        h_fwd.remove()
        h_bwd.remove()

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(rgb_image.size, Image.BILINEAR)
    heat = np.array(cam_img).astype(np.float32) / 255.0
    heat_rgb = _heat_colormap(heat)

    base = np.array(rgb_image.resize(rgb_image.size)).astype(np.float32)
    overlay = np.clip(0.55 * base + 0.45 * heat_rgb.astype(np.float32), 0, 255).astype(np.uint8)

    raw = CLASS_NAMES[pred_idx]
    plant, condition = pretty(raw)
    return Image.fromarray(overlay), plant, condition, confidence, None


def run_gradcam(image, selected_engine):
    if image is None:
        return None, "Upload a leaf image, then run Grad-CAM to see where the model is looking."

    overlay, plant, condition, confidence, note = compute_gradcam(image, selected_engine)
    if note:
        gr.Warning(note)
        return None, note

    engine_label = selected_engine.split(" ", 1)[1] if " " in selected_engine else selected_engine
    gr.Info(f"🔥 Grad-CAM ready — {engine_label} focused on the highlighted regions.")
    caption = (
        f"**{plant} · {condition}** ({confidence:.1f}% confidence, {engine_label}) — "
        f"brighter red/yellow = regions the model weighed most heavily for this diagnosis."
    )
    return overlay, caption


# -------------------------------------------------------
# 8. DATASET GALLERY
# -------------------------------------------------------
GALLERY_DIR = "PlantVillage_dataset"
GALLERY_ZIP = "PlantVillage_dataset.zip"
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp")
GALLERY_MAX_PER_LOAD = 36  # matches ~36 images per class folder in your dataset


def _ensure_gallery_extracted():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(app_dir, GALLERY_DIR)
    zip_path = os.path.join(app_dir, GALLERY_ZIP)

    if os.path.isdir(target) and os.listdir(target):
        return

    if not os.path.isfile(zip_path):
        return

    print(f"📦 Extracting {GALLERY_ZIP} → {GALLERY_DIR}/ ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(app_dir)
        print("✅ Gallery dataset extracted.")
    except Exception as e:
        print(f"⚠️ Failed to extract {GALLERY_ZIP}: {e}")


_ensure_gallery_extracted()


def _gallery_root():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), GALLERY_DIR)
    if os.path.isdir(base):
        entries = [e for e in os.listdir(base) if not e.startswith(".")]
        if len(entries) == 1:
            nested = os.path.join(base, entries[0])
            if os.path.isdir(nested) and any(
                os.path.isdir(os.path.join(nested, c)) for c in os.listdir(nested)
            ):
                return nested
        return base
    return None


def fetch_gallery_samples(species_filter, class_filter=None, n=36):
    """species_filter: substring match on folder name (e.g. 'Apple').
    class_filter: exact pretty class label (e.g. 'Apple · Apple scab') — takes priority when set."""
    root = _gallery_root()
    if root is None:
        return None, (
            f"Couldn't find the `{GALLERY_DIR}/` folder next to app.py. "
            f"Make sure you uploaded it to the Space repo root."
        )

    all_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    class_dirs = all_dirs

    if class_filter and class_filter != "All classes":
        raw_target = CLASS_LABEL_TO_RAW.get(class_filter)
        if raw_target:
            class_dirs = [d for d in all_dirs if d == raw_target or d.replace(" ", "_") == raw_target]
        if not class_dirs:
            # fallback: loose match ignoring punctuation/underscore-count differences
            norm_target = raw_target.lower().replace(" ", "").replace("_", "").replace(",", "") if raw_target else ""
            class_dirs = [d for d in all_dirs if d.lower().replace(" ", "").replace("_", "").replace(",", "") == norm_target]
    elif species_filter and species_filter != "All species":
        class_dirs = [d for d in all_dirs if species_filter.lower().replace(" ", "") in d.lower().replace("_", "").replace(",", "")]

    if not class_dirs:
        return None, "No matching class folders found for that filter."

    pool = []
    for d in class_dirs:
        folder = os.path.join(root, d)
        try:
            files = sorted(f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTS))
        except FileNotFoundError:
            continue
        for f in files:
            pool.append(os.path.join(folder, f))

    if not pool:
        return None, "No images found inside the matching class folder(s)."

    # A single exact class: show every image in that folder (up to n).
    # Species / all-species: shuffle across the matching folders for variety.
    if not (class_filter and class_filter != "All classes"):
        random.shuffle(pool)

    chosen = pool[:n]

    samples = []
    for path in chosen:
        try:
            samples.append(Image.open(path).convert("RGB"))
        except Exception:
            continue

    if not samples:
        return None, "Found files but couldn't open any as images — check the folder contents."
    return samples, len(pool)


def load_gallery(species_choice, class_choice):
    imgs, info = fetch_gallery_samples(species_choice, class_choice, n=GALLERY_MAX_PER_LOAD)
    if imgs is None:
        gr.Warning(f"⚠️ {info}")
        return [], f"⚠️ {info}"
    label = class_choice if (class_choice and class_choice != "All classes") else species_choice
    total_available = info
    shown = len(imgs)
    more_note = f" ({total_available} total in this filter — scroll to see them all)" if total_available > shown else ""
    gr.Info(f"🗂 Loaded {shown} images for {label}.")
    return imgs, f"Showing {shown} local samples for **{label}** from **{GALLERY_DIR}/**{more_note} — click a thumbnail, then hit *Send to Diagnose*."


def classes_for_species(species_choice):
    """Narrows the exact-class dropdown to only the classes belonging to the
    selected species — e.g. picking 'Apple' leaves only 'Apple · Apple scab',
    'Apple · Black rot', 'Apple · Cedar apple rust', 'Apple · healthy'."""
    if not species_choice or species_choice == "All species":
        matches = CLASS_LABELS
    else:
        matches = [label for label in CLASS_LABELS if label.split(" · ", 1)[0] == species_choice]
    return gr.Dropdown(choices=["All classes"] + matches, value="All classes")


# -------------------------------------------------------
# 9. STATIC HTML BLOCKS — leaderboard / about / species grid
# -------------------------------------------------------
def build_leaderboard_html():
    rows = ""
    for i, m in enumerate(LEADERBOARD):
        bar_pct = (m["acc"] - 90) / (100 - 90) * 100
        bar_pct = max(4, min(100, bar_pct))
        top3 = m["rank"] in ("🥇", "🥈", "🥉")
        rows += f"""
        <div class="lb-row {'lb-row-top' if top3 else ''}" style="animation-delay:{i*0.05:.2f}s;">
            <span class="lb-rank">{m['rank']}</span>
            <div class="lb-name-col">
                <span class="lb-name">{m['name']}</span>
                <span class="lb-type">{m['type']}</span>
            </div>
            <div class="lb-bar-track">
                <div class="lb-bar-fill" style="--target:{bar_pct}%;"></div>
            </div>
            <span class="lb-acc">{m['acc']:.2f}%</span>
            <span class="lb-loss">loss {m['loss']}</span>
        </div>"""
    return f"""
    <div class="glass-panel lb-panel">
        <p class="eyebrow">Trained on PlantVillage · 38 classes · 10 epochs each</p>
        <h2 class="panel-title">Model Leaderboard</h2>
        <div class="lb-header-row">
            <span></span><span>Model</span><span>Validation accuracy</span><span></span><span></span>
        </div>
        {rows}
        <p class="lb-footnote">Hypothesis under test: Vision Transformers generalize better than CNNs under Gaussian noise, thanks to global attention. Try it yourself in the 🌪️ Robustness Lab tab.</p>
    </div>
    """


def build_species_grid_html():
    chips = "".join(f'<span class="species-chip">{s}</span>' for s in SPECIES)
    return f"""
    <div class="glass-panel">
        <p class="eyebrow">14 crop species · 38 plant–disease classes</p>
        <h2 class="panel-title">Coverage</h2>
        <div class="species-grid">{chips}</div>
    </div>
    """


ABOUT_HTML = f"""
<div class="glass-panel about-panel">
    <p class="eyebrow">Research project</p>
    <h2 class="panel-title">How this system works</h2>

    <div class="about-grid">
        <div class="about-card">
            <span class="about-icon">🧪</span>
            <h3>Transfer learning</h3>
            <p>Every architecture starts from ImageNet weights. Backbones are frozen, classification heads are
            replaced with a fresh {NUM_CLASSES}-way layer, then fine-tuned on PlantVillage.</p>
        </div>
        <div class="about-card">
            <span class="about-icon">🌪️</span>
            <h3>Noise robustness</h3>
            <p>Each model is stress-tested with Gaussian noise to simulate poor lighting and low-quality
            phone cameras. Try it live in the <b>Robustness Lab</b> tab — drag the σ slider and compare.</p>
        </div>
        <div class="about-card">
            <span class="about-icon">⚔️</span>
            <h3>Ensemble consensus</h3>
            <p>The <b>Compare Engines</b> panel on the Diagnose tab runs all 8 live models on one image and
            surfaces where they agree — a quick sanity check beyond any single architecture's opinion.</p>
        </div>
        <div class="about-card">
            <span class="about-icon">⚙️</span>
            <h3>8 live engines</h3>
            <p>Every engine in the dropdown is downloaded on-demand from the
            <code>{HF_REPO}</code> Hub repo and cached in memory after first use.</p>
        </div>
        <div class="about-card">
            <span class="about-icon">🔥</span>
            <h3>Grad-CAM explainability</h3>
            <p>The <b>Grad-CAM</b> tab shows exactly which pixels a CNN weighed most heavily for its
            diagnosis — bright red/yellow overlay over the leaf. Works for all engines except the
            Swin Transformer, whose attention mechanism needs a different technique.</p>
        </div>
    </div>

    <div class="about-pipeline">
        <p class="eyebrow" style="margin-bottom:14px;">Pipeline</p>
        <div class="pipeline-track">
            <div class="pipeline-step"><span>01</span>Upload leaf</div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step"><span>02</span>Resize + normalize</div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step"><span>03</span>Selected engine infers</div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step"><span>04</span>Softmax + top-3</div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step"><span>05</span>Diagnostic report</div>
        </div>
    </div>

    <div class="about-section">
        <p class="eyebrow">Head-to-head research</p>
        <h3 class="about-section-title">8 architectures compared</h3>
        <table class="about-table">
            <thead>
                <tr><th>#</th><th>Model</th><th>Type</th><th>Input</th><th>Optimizer</th><th>LR</th></tr>
            </thead>
            <tbody>
                <tr><td>1</td><td>EfficientNet-B0</td><td>CNN (Baseline)</td><td>224×224</td><td>Adam</td><td><code>0.001</code></td></tr>
                <tr><td>2</td><td>Swin Transformer</td><td>Vision Transformer</td><td>224×224</td><td>Adam</td><td><code>5e-5</code></td></tr>
                <tr><td>3</td><td>ResNet-50</td><td>CNN</td><td>224×224</td><td>Adam</td><td><code>0.001</code></td></tr>
                <tr><td>4</td><td>DenseNet-121</td><td>CNN</td><td>224×224</td><td>Adam</td><td><code>0.0005</code></td></tr>
                <tr><td>5</td><td>Inception-V3</td><td>CNN</td><td>299×299</td><td>Adam</td><td><code>0.001</code></td></tr>
                <tr><td>6</td><td>MobileNet-V3 Large</td><td>Lightweight CNN</td><td>224×224</td><td>Adam</td><td><code>0.001</code></td></tr>
                <tr><td>7</td><td>VGG-16</td><td>Classic Deep CNN</td><td>224×224</td><td>Adam</td><td><code>0.0001</code></td></tr>
                <tr><td>8</td><td>ConvNeXt-Tiny</td><td>Modern CNN</td><td>224×224</td><td>AdamW</td><td><code>0.001</code></td></tr>
            </tbody>
        </table>
        <p style="color:var(--text-low); font-size:.8em; margin-top:12px; line-height:1.6;">
            All eight use <b style="color:var(--text-mid);">transfer learning</b> — pre-trained on ImageNet, with frozen
            backbones and fine-tuned classification heads, trained for 10 epochs each on a Kaggle GPU (CUDA).
        </p>
    </div>

    <div class="about-section">
        <div class="about-two-col">
            <div>
                <p class="eyebrow">Training data</p>
                <h3 class="about-section-title">PlantVillage dataset</h3>
                <ul class="about-list">
                    <li><b style="color:var(--text-hi);">38</b> plant–disease classes, RGB images, PyTorch <code>ImageFolder</code> format</li>
                    <li><b style="color:var(--text-hi);">80 / 20</b> train / validation split, batch size 32</li>
                    <li>14 crop species — Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato</li>
                    <li>Normalized with standard ImageNet mean / std, resized per-model (299×299 for Inception-V3, 224×224 for the rest)</li>
                </ul>
            </div>
            <div>
                <p class="eyebrow">Methodology</p>
                <h3 class="about-section-title">How each model was built</h3>
                <ul class="about-list">
                    <li>Download ImageNet pre-trained weights, freeze the backbone</li>
                    <li>Replace the classification head with a fresh 38-way <code>nn.Linear</code> layer</li>
                    <li>Train the head with <code>CrossEntropyLoss</code> for 10 epochs, tracking loss and accuracy every epoch</li>
                    <li>Stress-test with Gaussian noise (σ = 0.2) to simulate poor lighting / low-quality cameras</li>
                    <li>Explain predictions with Grad-CAM heatmaps over the last conv feature map</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="about-section">
        <p class="eyebrow">Final numbers</p>
        <h3 class="about-section-title">Results summary</h3>
        <table class="about-table">
            <thead><tr><th>Model</th><th>Best val. accuracy</th><th>Final train loss</th></tr></thead>
            <tbody>
                <tr><td>🥇 ConvNeXt-Tiny</td><td><span class="about-badge">99.13%</span></td><td><code>0.0236</code></td></tr>
                <tr><td>🥈 Swin Transformer</td><td><span class="about-badge">98.10%</span></td><td><code>0.0839</code></td></tr>
                <tr><td>🥉 VGG-16</td><td><span class="about-badge">97.57%</span></td><td><code>0.0323</code></td></tr>
                <tr><td>ResNet-50</td><td><span class="about-badge">97.01%</span></td><td><code>0.0957</code></td></tr>
                <tr><td>DenseNet-121</td><td><span class="about-badge">96.74%</span></td><td><code>0.2030</code></td></tr>
                <tr><td>MobileNet-V3</td><td><span class="about-badge">96.69%</span></td><td><code>0.0808</code></td></tr>
                <tr><td>EfficientNet-B0</td><td><span class="about-badge">96.54%</span></td><td><code>0.1466</code></td></tr>
                <tr><td>Inception-V3</td><td><span class="about-badge">93.60%</span></td><td><code>0.5975</code></td></tr>
            </tbody>
        </table>
        <p style="color:var(--text-low); font-size:.8em; margin-top:12px; line-height:1.6;">
            Hypothesis under test: <i>Vision Transformers are more robust to image noise than CNNs, thanks to their
            global attention mechanism.</i> See the full comparison on the 🏆 Leaderboard tab, or run it live yourself
            in 🌪️ Robustness Lab.
        </p>
    </div>

    <div class="about-section">
        <p class="eyebrow">Takeaways</p>
        <h3 class="about-section-title">Key insights</h3>
        <div class="about-insight-card">
            <b>🥇 ConvNeXt-Tiny wins overall</b>
            <p>The highest accuracy of the batch — a modern CNN that borrows design ideas from Vision Transformers, combining conv efficiency with transformer-style training tricks.</p>
        </div>
        <div class="about-insight-card">
            <b>👁️ Swin Transformer is the noise-robustness champion</b>
            <p>Its global attention mechanism keeps predictions more stable under Gaussian noise than traditional CNNs, supporting the project's core hypothesis.</p>
        </div>
        <div class="about-insight-card">
            <b>📱 MobileNet-V3 is the pick for edge deployment</b>
            <p>Its lightweight footprint makes it the most practical choice for IoT devices and mobile phones in the field.</p>
        </div>
        <div class="about-insight-card">
            <b>📦 VGG-16 still holds up</b>
            <p>Despite being the oldest architecture here, it performs competitively on a well-structured dataset like PlantVillage.</p>
        </div>
        <div class="about-insight-card">
            <b>🔥 Grad-CAM confirms the story</b>
            <p>Heatmaps show the ViT-based model focusing holistically on overall leaf structure, while CNNs sometimes latch onto smaller, irrelevant texture patterns.</p>
        </div>
    </div>

    <div class="about-section">
        <p class="eyebrow">Credits</p>
        <h3 class="about-section-title">Acknowledgements</h3>
        <div class="about-links">
            <span class="about-link-chip">📊 PlantVillage Dataset — Abdallah Ali</span>
            <span class="about-link-chip">🔥 PyTorch — Deep Learning Framework</span>
            <span class="about-link-chip">🧩 pytorch-grad-cam — Explainability</span>
            <span class="about-link-chip">🎛️ Gradio — ML Demo Framework</span>
            <span class="about-link-chip">☁️ Kaggle — Free GPU Environment</span>
        </div>
    </div>
</div>
"""

# -------------------------------------------------------
# 10. CUSTOM CSS — "Bio-Scanner" design system v3 (Ultra)
# -------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-deep: #060b0a;
    --bg-panel: #0d1614;
    --bg-panel-2: #101c19;
    --line: rgba(57,255,157,.14);
    --leaf: #39ff9d;
    --leaf-dim: #1c8a5c;
    --text-hi: #eaf7f0;
    --text-mid: #9db3ab;
    --text-low: #5e746c;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scrollbar-color: var(--leaf) var(--bg-panel); scrollbar-width: auto; }

::-webkit-scrollbar { width: 14px; height: 14px; }
::-webkit-scrollbar-track { background: var(--bg-panel); border-radius: 10px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--leaf-dim), var(--leaf));
    border-radius: 10px; border: 3px solid var(--bg-panel);
    box-shadow: 0 0 8px rgba(57,255,157,.5);
}
::-webkit-scrollbar-thumb:hover { background: var(--leaf); box-shadow: 0 0 14px rgba(57,255,157,.8); }

/* Any element that scrolls (the gallery especially) gets a visible themed
   scrollbar too — Firefox via scrollbar-color, Chromium/Safari via ::-webkit-*. */
[data-testid="gallery"] { scrollbar-color: var(--leaf) var(--bg-panel); scrollbar-width: auto; }
[data-testid="gallery"]::-webkit-scrollbar { width: 14px; }
[data-testid="gallery"]::-webkit-scrollbar-track { background: var(--bg-panel); border-radius: 10px; }
[data-testid="gallery"]::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--leaf-dim), var(--leaf));
    border-radius: 10px; border: 3px solid var(--bg-panel);
    box-shadow: 0 0 10px rgba(57,255,157,.6);
}

.gradio-container {
    font-family: 'Space Grotesk', 'Segoe UI', sans-serif !important;
    background:
        radial-gradient(circle at 15% 0%, rgba(57,255,157,.08) 0%, transparent 45%),
        radial-gradient(circle at 85% 100%, rgba(57,255,157,.05) 0%, transparent 50%),
        var(--bg-deep) !important;
    color: var(--text-hi) !important;
    position: relative;
    overflow-x: hidden;
    animation: appIn .5s ease both;
}
@keyframes appIn { from { opacity: 0; } to { opacity: 1; } }
footer { display:none !important; }
gradio-app { background: var(--bg-deep) !important; }

/* ---------- animated grid + floating leaf particles ---------- */
.gradio-container::before {
    content: "";
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(57,255,157,.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(57,255,157,.045) 1px, transparent 1px);
    background-size: 46px 46px;
    mask-image: radial-gradient(ellipse 80% 60% at 50% 0%, black 20%, transparent 75%);
    pointer-events: none; z-index: 0;
}
.particle-field {
    position: fixed; inset: 0; pointer-events: none; overflow: hidden; z-index: 0;
}
.particle {
    position: absolute; bottom: -10%;
    font-size: 22px; opacity: 0; filter: drop-shadow(0 0 6px rgba(57,255,157,.35));
    animation: floatUp linear infinite;
}
@keyframes floatUp {
    0%   { transform: translateY(0) translateX(0) rotate(0deg); opacity: 0; }
    8%   { opacity: .5; }
    92%  { opacity: .35; }
    100% { transform: translateY(-115vh) translateX(var(--drift, 40px)) rotate(340deg); opacity: 0; }
}

/* ---------- hero ---------- */
@keyframes heroGlow { 0%,100%{ opacity:.55; transform:translateX(-6%);} 50%{ opacity:1; transform:translateX(6%);} }
@keyframes fadeUp { from{ opacity:0; transform:translateY(14px);} to{ opacity:1; transform:translateY(0);} }
.hero-header {
    position: relative; overflow: hidden;
    background: linear-gradient(160deg, #08130f 0%, #0c1e17 55%, #0a1512 100%);
    border: 1px solid var(--line); border-radius: 18px;
    padding: 34px 40px; margin-bottom: 22px;
    animation: fadeUp .6s ease both; z-index:1;
}
.hero-header::after {
    content: ""; position: absolute; top: -40%; left: -10%; width: 60%; height: 220%;
    background: radial-gradient(closest-side, rgba(57,255,157,.18), transparent 70%);
    animation: heroGlow 6s ease-in-out infinite; pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'JetBrains Mono', monospace; font-size: .72em; letter-spacing: 2px; text-transform: uppercase;
    color: var(--leaf); background: rgba(57,255,157,.08); border: 1px solid var(--line);
    padding: 5px 12px; border-radius: 30px; margin-bottom: 14px;
}
.hero-eyebrow .pulse-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--leaf);
    box-shadow: 0 0 8px 2px rgba(57,255,157,.8); animation: pulseDot 1.6s ease-in-out infinite;
}
@keyframes pulseDot { 0%,100%{ opacity:1; transform:scale(1);} 50%{ opacity:.4; transform:scale(.7);} }
.hero-header h1 { color:#fff !important; margin:0; font-size:2.5em; font-weight:700; letter-spacing:-.5px; position:relative; z-index:1; }
.hero-header h1 span { color: var(--leaf); }
.hero-header p { color: var(--text-mid) !important; margin:10px 0 0; font-size:1.05em; max-width:620px; position:relative; z-index:1; }
.hero-stats { display:flex; gap:28px; margin-top:22px; position:relative; z-index:1; flex-wrap:wrap; }
.hero-stat { display:flex; flex-direction:column; animation: fadeUp .5s ease both; }
.hero-stat:nth-child(1) { animation-delay: .15s; }
.hero-stat:nth-child(2) { animation-delay: .25s; }
.hero-stat:nth-child(3) { animation-delay: .35s; }
.hero-stat:nth-child(4) { animation-delay: .45s; }
.hero-stat b { font-size:1.5em; color:#fff; font-family:'JetBrains Mono',monospace; }
.hero-stat span { font-size:.75em; color: var(--text-low); text-transform:uppercase; letter-spacing:1px; margin-top:2px; }

/* ---------- tabs ---------- */
.tabs { border: none !important; z-index:1; position:relative; }
.tab-nav { gap: 6px !important; border-bottom: 1px solid var(--line) !important; }
.tab-nav button {
    font-family:'JetBrains Mono', monospace !important; font-size:.85em !important;
    color: var(--text-mid) !important; border-radius: 10px 10px 0 0 !important;
    background: transparent !important; border: none !important; padding: 10px 18px !important;
    position: relative; transition: color .2s ease !important;
}
.tab-nav button.selected {
    color: var(--leaf) !important; background: rgba(57,255,157,.08) !important;
    box-shadow: inset 0 -2px 0 var(--leaf) !important;
}
.tab-nav button.selected::after {
    content: ""; position: absolute; left: 12px; right: 12px; bottom: -1px; height: 2px;
    background: var(--leaf); border-radius: 2px;
    animation: tabIn .25s cubic-bezier(.22,1,.36,1) both;
}
@keyframes tabIn { from { transform: scaleX(0); opacity:0; } to { transform: scaleX(1); opacity:1; } }
.tabitem { animation: panelIn .35s ease both; }
/* NOTE: intentionally opacity-only. A transform here (even with fill-mode:
   both/forwards) permanently turns .tabitem into the positioning anchor for
   any fixed-position popup inside it — e.g. Dropdown option lists — which is
   what caused those to render in the wrong place. Do not add transform back. */
@keyframes panelIn { from { opacity:0; } to { opacity:1; } }

/* ---------- panels / cards (with cursor-glow) ---------- */
.control-card, .glass-panel {
    background: linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-panel-2) 100%);
    border: 1px solid var(--line); border-radius: 16px; padding: 22px; position: relative; z-index: 1;
    --mx: 50%; --my: 50%;
    transition: border-color .25s ease, transform .25s ease, box-shadow .25s ease;
}
.control-card::before, .glass-panel::before {
    content:""; position:absolute; inset:0; border-radius:inherit;
    background: radial-gradient(320px circle at var(--mx) var(--my), rgba(57,255,157,.09), transparent 60%);
    opacity:0; transition:opacity .35s ease; pointer-events:none;
}
.control-card:hover::before, .glass-panel:hover::before { opacity:1; }
.control-card:hover, .glass-panel:hover { border-color: rgba(57,255,157,.28); box-shadow: 0 14px 34px -18px rgba(0,0,0,.6); }
.panel-title { color:#fff; font-size:1.5em; font-weight:700; margin:2px 0 18px; }
.section-label {
    font-family: 'JetBrains Mono', monospace; font-size: .78em; text-transform: uppercase; letter-spacing: 1.5px;
    color: var(--leaf); margin: 0 0 14px !important; display:flex; align-items:center; gap:8px;
}
.section-label::before { content:""; width:16px; height:2px; background:var(--leaf); display:inline-block; }

/* ---------- gradio component skinning ---------- */
button.primary, button.secondary {
    border-radius: 12px !important; font-weight: 600 !important;
    transition: transform .15s cubic-bezier(.22,1,.36,1), box-shadow .15s ease, border-color .15s ease, color .15s ease !important;
    position: relative !important; overflow: hidden !important;
}
/* Safety net: make sure any dropdown/select option list always floats above
   everything else and is never clipped, regardless of Gradio version internals. */
ul[role="listbox"], .options, [data-testid="dropdown"] ul {
    z-index: 99999 !important;
}
button.primary {
    background: linear-gradient(135deg, #39ff9d, #12b76a) !important; color: #06110c !important; border: none !important;
}
button.primary:hover { transform: translateY(-2px); box-shadow: 0 8px 22px -6px rgba(57,255,157,.55) !important; }
button.primary:active, button.secondary:active { transform: translateY(0) scale(.96) !important; }
button.secondary { background: rgba(255,255,255,.03) !important; border: 1px solid var(--line) !important; color: var(--text-mid) !important; }
button.secondary:hover { transform: translateY(-2px); border-color: var(--leaf) !important; color: var(--leaf) !important; }
button.primary::before {
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(120deg, transparent 30%, rgba(255,255,255,.35) 50%, transparent 70%);
    transform: translateX(-120%); pointer-events:none;
}
button.primary:hover::before { animation: sheen 1.1s ease; }
@keyframes sheen { to { transform: translateX(120%); } }

/* button ripple (positioned via JS on click) */
.btn-ripple {
    position:absolute; border-radius:50%; background:rgba(255,255,255,.55);
    transform:scale(0); animation:rippleAnim .6s ease-out; pointer-events:none;
    width:10px; height:10px; margin-left:-5px; margin-top:-5px;
}
@keyframes rippleAnim { to { transform:scale(22); opacity:0; } }

.image-container, [data-testid="image"] {
    border-radius: 14px !important; border: 1.5px dashed var(--line) !important; transition: border-color .25s ease, box-shadow .25s ease;
}
.image-container:hover, [data-testid="image"]:hover { border-color: var(--leaf) !important; box-shadow: 0 0 0 4px rgba(57,255,157,.06); }

/* slider skin */
input[type="range"] { accent-color: var(--leaf) !important; }

/* ---------- idle / error card ---------- */
.idle-card {
    text-align: center; padding: 70px 24px; min-height: 380px;
    display:flex; flex-direction:column; align-items:center; justify-content:center; animation: fadeUp .5s ease both;
}
.idle-orbit { position: relative; width: 120px; height: 120px; margin-bottom: 22px; }
.idle-ring { position: absolute; inset: 0; border-radius: 50%; border: 1px solid var(--line); }
.ring-1 { animation: spin 8s linear infinite; }
.ring-2 { inset: 14px; animation: spin 6s linear infinite reverse; border-color: rgba(57,255,157,.25); }
.ring-3 { inset: 28px; animation: spin 4s linear infinite; border-color: rgba(57,255,157,.4); }
@keyframes spin { to { transform: rotate(360deg); } }
.idle-emoji { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size: 2.3em; filter: drop-shadow(0 0 12px rgba(57,255,157,.4)); }
.idle-card h2 { color: var(--text-hi); font-weight: 600; margin: 0; font-size: 1.3em; }
.idle-sub { color: var(--text-low); margin-top: 8px; font-size: .92em; }
.error-card h2 { color: #ff5470; }

/* ---------- skeleton loading card ---------- */
.skeleton-card { padding: 6px 0; animation: fadeUp .35s ease both; }
.skel-top { display:flex; align-items:center; gap:12px; margin-bottom:22px; }
.skel-spinner { width:22px; height:22px; border-radius:50%; border:2.5px solid rgba(57,255,157,.15); border-top-color: var(--leaf); animation: spin .8s linear infinite; }
.skel-live-label { font-family:'JetBrains Mono',monospace; font-size:.85em; color: var(--leaf); letter-spacing:.5px; }
.skel-dots span { animation: dotBlink 1.2s infinite; opacity:0; }
.skel-dots span:nth-child(2) { animation-delay:.2s; }
.skel-dots span:nth-child(3) { animation-delay:.4s; }
@keyframes dotBlink { 0%,100%{opacity:0;} 50%{opacity:1;} }
.skel-line, .skel-block {
    border-radius:8px; margin-bottom:14px;
    background: linear-gradient(90deg, rgba(255,255,255,.035) 25%, rgba(255,255,255,.09) 37%, rgba(255,255,255,.035) 63%);
    background-size: 400% 100%; animation: shimmer 1.4s ease infinite;
}
.skel-line { height:14px; }
.skel-title { height:30px; width:55%; }
.skel-block { height:70px; margin:18px 0; }
@keyframes shimmer { 0%{ background-position: 100% 0; } 100%{ background-position: 0 0; } }

/* ---------- result card ---------- */
.result-wrap { position: relative; animation: fadeUp .45s ease both; overflow: hidden; }
.scan-line {
    position: absolute; left:0; right:0; top:0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--leaf), transparent);
    animation: scanDown 1.8s ease-in-out 1; box-shadow: 0 0 12px 2px rgba(57,255,157,.6);
}
@keyframes scanDown { 0%{ top:0; opacity:1;} 100%{ top:100%; opacity:0;} }
.result-header { display:flex; justify-content:space-between; align-items:center; padding-bottom: 18px; margin-bottom: 22px; border-bottom: 1px solid var(--line); flex-wrap: wrap; gap: 10px; }
.status-pill {
    display:inline-flex; align-items:center; gap:8px;
    font-family:'JetBrains Mono', monospace; font-weight:700; font-size:.85em; letter-spacing:.5px;
    color: var(--accent); background: var(--accent-soft); border: 1px solid var(--accent); padding: 8px 16px; border-radius: 30px;
}
.status-dot { width:8px; height:8px; border-radius:50%; background: var(--accent); box-shadow: 0 0 8px 2px var(--accent); animation: pulseDot 1.5s ease-in-out infinite; }
.engine-chip { font-family:'JetBrains Mono', monospace; font-size:.78em; color: var(--text-mid); background: rgba(255,255,255,.03); border:1px solid var(--line); padding: 7px 14px; border-radius: 8px; }
.id-block { animation: fadeUp .5s ease both; margin-bottom: 18px; }
.eyebrow { font-family:'JetBrains Mono', monospace; font-size:.72em; text-transform:uppercase; letter-spacing:1.5px; color: var(--text-low); margin: 0 0 4px; }
.crop-name { color:#fff; margin:0; font-size:2.3em; font-weight:700; letter-spacing:-.5px; }
.condition-name { margin:0; font-size:1.5em; font-weight:700; }
.confidence-block { animation: fadeUp .5s ease both; margin: 26px 0; }
.confidence-top { display:flex; justify-content:space-between; margin-bottom:8px; }
.conf-label { font-family:'JetBrains Mono', monospace; font-size:.78em; text-transform:uppercase; letter-spacing:1px; color:var(--text-mid); }
.conf-value { font-weight:800; font-family:'JetBrains Mono', monospace; }
.conf-track { width:100%; background: rgba(255,255,255,.05); border-radius:20px; height:10px; overflow:hidden; }
.conf-fill { height:100%; border-radius:20px; width:0%; animation: fillBar 1.1s cubic-bezier(.22,1,.36,1) .3s forwards; }
@keyframes fillBar { to { width: var(--target); } }
.action-block { border-left: 4px solid; border-radius: 10px; padding: 16px 18px; animation: fadeUp .5s ease both; margin-bottom: 24px; }
.action-block strong { color: var(--text-hi); font-size: .92em; }
.action-block p { color: var(--text-mid); margin: 8px 0 0; line-height: 1.6; font-size: .92em; }
.rank-block { animation: fadeUp .5s ease both; }
.rank-row { display:grid; grid-template-columns: 26px 1fr 100px 52px; align-items:center; gap: 12px; padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,.04); animation: fadeUp .4s ease both; font-size: .86em; }
.rank-row:last-child { border-bottom:none; }
.rank-idx { font-family:'JetBrains Mono', monospace; color: var(--text-low); }
.rank-label { color: var(--text-mid); }
.rank-row-top .rank-label { color: var(--text-hi); font-weight: 600; }
.rank-bar-track { background: rgba(255,255,255,.05); border-radius: 10px; height: 6px; overflow:hidden; }
.rank-bar-fill { height:100%; width:0%; border-radius:10px; animation: fillBar 1s cubic-bezier(.22,1,.36,1) .5s forwards; }
.rank-pct { font-family:'JetBrains Mono', monospace; text-align:right; }

/* copy button inside result card */
.copy-btn {
    margin-top: 22px; display:inline-flex; align-items:center; gap:8px; cursor:pointer;
    font-family:'JetBrains Mono', monospace; font-size:.78em; letter-spacing:.5px;
    background: rgba(255,255,255,.03); border:1px solid var(--line); color: var(--text-mid);
    padding: 9px 16px; border-radius: 10px; animation: fadeUp .5s ease both;
    transition: border-color .2s ease, color .2s ease, transform .15s ease;
}
.copy-btn:hover { border-color: var(--leaf); color: var(--leaf); transform: translateY(-2px); }
.copy-btn:active { transform: translateY(0) scale(.96); }
.copy-btn.copied { border-color: var(--leaf); color: var(--leaf); background: rgba(57,255,157,.08); }
.copy-btn span::before { content: "📋 "; }
.copy-btn.copied span::before { content: "✓ "; }

/* ---------- compare-all-engines panel ---------- */
.consensus-banner {
    display:flex; align-items:center; justify-content:space-between; gap:20px;
    border: 1px solid var(--c-accent); background: color-mix(in srgb, var(--c-accent) 10%, transparent);
    border-radius: 14px; padding: 20px 24px; margin-bottom: 22px; animation: fadeUp .45s ease both; flex-wrap: wrap;
}
.consensus-banner h2 { margin: 4px 0 0; font-size: 1.6em; font-weight: 700; }
.consensus-ring {
    width: 64px; height: 64px; border-radius: 50%; flex-shrink: 0;
    background: conic-gradient(var(--c-accent) calc(var(--pct)*1%), rgba(255,255,255,.08) 0);
    display:flex; align-items:center; justify-content:center; position: relative;
    animation: ringIn .8s cubic-bezier(.22,1,.36,1) both;
}
.consensus-ring::before { content:""; position:absolute; inset:6px; border-radius:50%; background: var(--bg-panel); }
.consensus-ring span { position:relative; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:.85em; }
@keyframes ringIn { from { transform: scale(.6); opacity:0; } to { transform: scale(1); opacity:1; } }
.engine-row {
    display:grid; grid-template-columns: 1.1fr 1.3fr 90px 52px 80px; align-items:center; gap: 12px;
    padding: 11px 0; border-bottom: 1px solid rgba(255,255,255,.04); animation: fadeUp .4s ease both; font-size: .85em;
}
.engine-row:last-child { border-bottom: none; }
.engine-row-name { color: var(--text-hi); font-weight: 600; }
.engine-row-name em { color: var(--text-low); font-style: normal; font-size: .85em; }
.engine-row-label { color: var(--text-mid); }
.engine-row-track { width: 100%; }
.agree-tag { font-family:'JetBrains Mono',monospace; font-size:.72em; color: var(--leaf); background: rgba(57,255,157,.1); border:1px solid var(--line); padding:3px 8px; border-radius:20px; text-align:center; }
.engine-row-agree { background: linear-gradient(90deg, rgba(57,255,157,.05), transparent); border-radius: 8px; }
.engine-row-error { color: #ff5470; grid-template-columns: 1fr 1fr; }
.engine-row-status { font-family:'JetBrains Mono',monospace; font-size:.8em; }

/* ---------- robustness lab ---------- */
.robustness-verdict {
    font-family:'JetBrains Mono',monospace; font-size:.85em; color: var(--text-mid);
    background: rgba(255,255,255,.02); border:1px solid var(--line); border-radius:12px;
    padding: 14px 18px; margin-top: 4px; animation: fadeUp .4s ease both; line-height: 1.6;
}

/* ---------- leaderboard ---------- */
.lb-panel { animation: fadeUp .5s ease both; }
.lb-header-row {
    display:grid; grid-template-columns: 32px 1fr 1fr 90px 90px; gap:12px;
    font-family:'JetBrains Mono',monospace; font-size:.7em; text-transform:uppercase; letter-spacing:1px;
    color: var(--text-low); padding: 6px 0 10px; border-bottom: 1px solid var(--line);
}
.lb-row {
    display:grid; grid-template-columns: 32px 1fr 1fr 90px 90px; gap:12px; align-items:center;
    padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,.04); animation: fadeUp .4s ease both;
    transition: background .2s ease;
}
.lb-row:hover { background: rgba(57,255,157,.03); }
.lb-row:last-child { border-bottom: none; }
.lb-row-top .lb-name { color: var(--leaf); }
.lb-rank { font-family:'JetBrains Mono',monospace; font-size:1.1em; text-align:center; color: var(--text-mid); }
.lb-name-col { display:flex; flex-direction:column; }
.lb-name { color:#fff; font-weight:600; font-size:.95em; }
.lb-type { color: var(--text-low); font-size:.75em; margin-top:2px; }
.lb-bar-track { background: rgba(255,255,255,.05); border-radius:10px; height:8px; overflow:hidden; }
.lb-bar-fill { height:100%; width:0%; border-radius:10px; background: linear-gradient(90deg,#1c8a5c,#39ff9d); animation: fillBar 1.2s cubic-bezier(.22,1,.36,1) .2s forwards; }
.lb-acc { font-family:'JetBrains Mono',monospace; font-weight:700; color: var(--text-hi); text-align:right; }
.lb-loss { font-family:'JetBrains Mono',monospace; font-size:.72em; color: var(--text-low); text-align:right; }
.lb-footnote { margin-top:22px; color: var(--text-low); font-size:.82em; line-height:1.6; border-top:1px solid var(--line); padding-top:16px; }

/* ---------- species grid ---------- */
.species-grid { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
.species-chip {
    font-family:'JetBrains Mono',monospace; font-size:.78em; color: var(--text-mid);
    background: rgba(57,255,157,.05); border: 1px solid var(--line); padding: 6px 14px; border-radius: 20px;
    transition: all .15s ease;
}
.species-chip:hover { color: var(--leaf); border-color: var(--leaf); transform: translateY(-2px) scale(1.04); }

/* ---------- about ---------- */
.about-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:16px; margin-top:10px; }
.about-card { background: rgba(255,255,255,.02); border:1px solid var(--line); border-radius:12px; padding:18px; transition: transform .15s ease, border-color .15s ease; }
.about-card:hover { transform: translateY(-3px); border-color: var(--leaf); }
.about-icon { font-size:1.6em; }
.about-card h3 { color:#fff; font-size:1em; margin:10px 0 6px; }
.about-card p { color: var(--text-mid); font-size:.85em; line-height:1.55; margin:0; }
.about-card code { color: var(--leaf); background: rgba(57,255,157,.08); padding:1px 6px; border-radius:4px; font-size:.9em; }
.about-pipeline { margin-top:26px; border-top:1px solid var(--line); padding-top:20px; }
.pipeline-track { display:flex; flex-wrap:wrap; align-items:center; gap:10px; }
.pipeline-step {
    font-family:'JetBrains Mono',monospace; font-size:.78em; color: var(--text-mid);
    background: rgba(255,255,255,.02); border:1px solid var(--line); border-radius:10px; padding:10px 14px;
    display:flex; flex-direction:column; gap:4px; min-width:120px; transition: transform .15s ease, border-color .15s ease;
}
.pipeline-step:hover { transform: translateY(-2px); border-color: var(--leaf); }
.pipeline-step span { color: var(--leaf); font-weight:700; }
.pipeline-arrow { color: var(--leaf-dim); font-size:1.2em; }

/* ---------- about: extra sections (architectures table, dataset, methodology, results, insights) ---------- */
.about-section { margin-top: 28px; border-top: 1px solid var(--line); padding-top: 22px; }
.about-section .eyebrow { margin-bottom: 6px; }
.about-section-title { color: #fff; font-size: 1.15em; font-weight: 700; margin: 0 0 14px; }
.about-table { width: 100%; border-collapse: collapse; font-size: .85em; }
.about-table th {
    font-family:'JetBrains Mono',monospace; font-size:.72em; text-transform:uppercase; letter-spacing:1px;
    color: var(--text-low); text-align:left; padding: 8px 10px; border-bottom: 1px solid var(--line);
}
.about-table td {
    padding: 10px; border-bottom: 1px solid rgba(255,255,255,.04); color: var(--text-mid); vertical-align: middle;
}
.about-table tr:hover td { background: rgba(57,255,157,.03); }
.about-table td:first-child, .about-table th:first-child { color: var(--text-hi); font-weight: 600; }
.about-table code { color: var(--leaf); background: rgba(57,255,157,.08); padding: 1px 6px; border-radius: 4px; font-size: .92em; }
.about-badge {
    display:inline-block; font-family:'JetBrains Mono',monospace; font-size:.72em;
    color: var(--leaf); background: rgba(57,255,157,.08); border: 1px solid var(--line);
    padding: 3px 10px; border-radius: 20px;
}
.about-two-col { display:grid; grid-template-columns: 1fr 1fr; gap: 22px; }
.about-list { list-style: none; padding: 0; margin: 0; }
.about-list li {
    color: var(--text-mid); font-size: .88em; line-height: 1.7; padding-left: 20px; position: relative;
}
.about-list li::before { content: "▸"; color: var(--leaf); position: absolute; left: 0; }
.about-insight-card {
    background: rgba(255,255,255,.02); border: 1px solid var(--line); border-left: 3px solid var(--leaf);
    border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; transition: transform .15s ease, border-color .15s ease;
}
.about-insight-card:hover { transform: translateX(3px); border-color: var(--leaf); }
.about-insight-card b { color: var(--text-hi); font-size: .9em; }
.about-insight-card p { color: var(--text-mid); font-size: .85em; line-height: 1.55; margin: 6px 0 0; }
.about-links { display:flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
.about-link-chip {
    font-family:'JetBrains Mono',monospace; font-size:.78em; color: var(--text-mid);
    background: rgba(255,255,255,.02); border: 1px solid var(--line); border-radius: 10px;
    padding: 8px 14px; text-decoration: none; transition: all .15s ease; display: inline-flex; align-items: center; gap: 6px;
}
.about-link-chip:hover { color: var(--leaf); border-color: var(--leaf); transform: translateY(-2px); }
@media (max-width: 720px) { .about-two-col { grid-template-columns: 1fr; } .about-table { font-size: .78em; } }

/* ---------- gallery ---------- */
.gallery-note { font-family:'JetBrains Mono',monospace; font-size:.8em; color: var(--text-mid); margin-top:10px; }
.gallery { animation: fadeUp .5s ease both; }
/* The gallery must own its scroll — the page must never grow because of it.
   Force max-height + overflow on the actual container Gradio renders, and let
   the inner grid grow freely inside it (that's what makes it scrollable). Use
   both the elem_classes selector and the data-testid selector, and mark the
   overflow rules !important so no later Gradio theme rule can silently win
   and swallow the scrollbar again. */
.gradio-container [data-testid="gallery"],
.gradio-container .gallery {
    max-height: 560px !important;
    overflow-y: scroll !important;
    overflow-x: hidden !important;
    overscroll-behavior: contain;
    scroll-behavior: smooth;
}
.gradio-container [data-testid="gallery"] .grid-wrap,
.gradio-container [data-testid="gallery"] > div,
.gradio-container .gallery .grid-wrap,
.gradio-container .gallery > div {
    max-height: none !important;
    overflow: visible !important;
}
.gallery-item, [data-testid="gallery"] .grid-wrap .thumbnail-item {
    border-radius: 12px !important; border: 1px solid var(--line) !important;
    transition: transform .2s cubic-bezier(.22,1,.36,1), box-shadow .2s ease, border-color .2s ease !important;
}
.gallery-item:hover, [data-testid="gallery"] .grid-wrap .thumbnail-item:hover {
    transform: translateY(-4px) scale(1.02) !important;
    border-color: var(--leaf) !important;
    box-shadow: 0 10px 26px -8px rgba(57,255,157,.45) !important;
    z-index: 2;
}

/* ---------- accordion (compare-engines panel) ---------- */
.gradio-container details {
    border: 1px solid var(--line) !important; border-radius: 14px !important; background: var(--bg-panel) !important;
    overflow: hidden;
}
.gradio-container summary {
    font-family:'JetBrains Mono', monospace !important; color: var(--leaf) !important; font-size:.85em !important;
    letter-spacing:.5px; padding: 4px 6px !important; cursor: pointer;
}

/* ---------- accessibility / motion ---------- */
button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {
    outline: 2px solid var(--leaf) !important; outline-offset: 2px !important;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .001ms !important; animation-iteration-count: 1 !important; }
}

@media (max-width: 640px) {
    .hero-header h1 { font-size: 1.7em; }
    .crop-name { font-size: 1.7em; }
    .rank-row { grid-template-columns: 20px 1fr 70px 44px; }
    .engine-row { grid-template-columns: 1fr; row-gap: 6px; }
    .lb-header-row, .lb-row { grid-template-columns: 24px 1fr 60px; }
    .lb-header-row span:nth-child(4), .lb-header-row span:nth-child(5),
    .lb-row .lb-bar-track, .lb-row .lb-loss { display:none; }
}
"""

PARTICLE_HTML = """
<div class="particle-field">
""" + "".join(
    f'<span class="particle" style="left:{p}%; animation-duration:{d}s; animation-delay:{delay}s; --drift:{drift}px;">🌿</span>'
    for p, d, delay, drift in [
        (6, 22, 0, 60), (18, 26, 4, -40), (32, 19, 2, 50), (47, 24, 7, -70),
        (61, 21, 1, 30), (74, 27, 5, -50), (86, 23, 3, 40), (94, 25, 8, -30),
    ]
) + "</div>"

# Injected once into <head>: global micro-interactions (button ripple + cursor glow)
HEAD_JS = """
<script>
window.addEventListener('DOMContentLoaded', function () {
  document.addEventListener('click', function (e) {
    var btn = e.target.closest('button.primary, button.secondary');
    if (!btn) return;
    var rect = btn.getBoundingClientRect();
    var ripple = document.createElement('span');
    ripple.className = 'btn-ripple';
    ripple.style.left = (e.clientX - rect.left) + 'px';
    ripple.style.top = (e.clientY - rect.top) + 'px';
    btn.appendChild(ripple);
    setTimeout(function () { ripple.remove(); }, 650);
  });

  document.addEventListener('mousemove', function (e) {
    document.querySelectorAll('.control-card, .glass-panel').forEach(function (card) {
      var r = card.getBoundingClientRect();
      if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom) {
        card.style.setProperty('--mx', (e.clientX - r.left) + 'px');
        card.style.setProperty('--my', (e.clientY - r.top) + 'px');
      }
    });
  });
});
</script>
"""

# -------------------------------------------------------
# 11. GRADIO UI
# -------------------------------------------------------
with gr.Blocks(css=CSS, title="Agro-Vision · Plant Disease AI", theme=gr.themes.Base(), head=HEAD_JS) as demo:

    gr.HTML(PARTICLE_HTML)

    gr.HTML("""
        <div class="hero-header">
            <span class="hero-eyebrow"><span class="pulse-dot"></span> Live multi-model inference</span>
            <h1>🌿 <span>Agro-Vision</span> Diagnostic System</h1>
            <p>Point your camera at a leaf, pick a neural network, and watch eight rival architectures
            race to a diagnosis — then cross-examine them against each other, throw noise at them to see
            who breaks first, and raid the training set they learned from, all without leaving this screen.</p>
            <div class="hero-stats">
                <div class="hero-stat"><b>38</b><span>Disease classes</span></div>
                <div class="hero-stat"><b>14</b><span>Crop species</span></div>
                <div class="hero-stat"><b>8</b><span>Architectures trained</span></div>
                <div class="hero-stat"><b>99.13%</b><span>Best val. accuracy</span></div>
            </div>
        </div>
    """)

    with gr.Tabs() as main_tabs:
        # ---------------- DIAGNOSE TAB ----------------
        with gr.Tab("🔬 Diagnose", id=0):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="control-card"):
                        gr.Markdown("### 🧠 Engine", elem_classes="section-label")
                        model_dd = gr.Dropdown(
                            choices=list(MODEL_CONFIG.keys()),
                            value="🏆 ConvNeXt-Tiny (Champion)",
                            show_label=False, container=False,
                        )
                        gr.Markdown("### 📸 Specimen", elem_classes="section-label")
                        img_input = gr.Image(type="pil", show_label=False)
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Reset", variant="secondary")
                            analyze_btn = gr.Button("🔍 Run Diagnosis", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Diagnostic Report", elem_classes="section-label")
                    out_html = gr.HTML(value=_idle_card(), elem_classes="glass-panel")

                    with gr.Accordion("⚔️ Compare all 8 engines on this image", open=False):
                        compare_btn = gr.Button("⚔️ Run Ensemble Comparison", variant="secondary")
                        compare_html = gr.HTML(value=_idle_card("Run the comparison to see every engine's verdict"))

            def _analyze_and_toast(image, engine):
                if image is None:
                    gr.Warning("Upload a leaf image first.")
                    return _idle_card("Awaiting specimen — upload a leaf image to begin")
                result = analyze_plant(image, engine)
                engine_label = engine.split(" ", 1)[1] if " " in engine else engine
                gr.Info(f"🔬 Diagnosis complete with {engine_label}.")
                return result

            analyze_btn.click(fn=lambda: _loading_card("Running inference"), inputs=None, outputs=out_html).then(
                fn=_analyze_and_toast, inputs=[img_input, model_dd], outputs=out_html
            )
            clear_btn.click(
                fn=lambda: (None, _idle_card(), _idle_card("Run the comparison to see every engine's verdict")),
                inputs=[], outputs=[img_input, out_html, compare_html],
            )
            compare_btn.click(
                fn=lambda: _loading_card("Waking up all 8 engines"), inputs=None, outputs=compare_html
            ).then(fn=compare_all_engines, inputs=[img_input], outputs=compare_html)

        # ---------------- ROBUSTNESS LAB TAB ----------------
        with gr.Tab("🌪️ Robustness Lab", id=1):
            gr.Markdown("### 🧪 Stress-test an engine with live Gaussian noise", elem_classes="section-label")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="control-card"):
                        rob_model_dd = gr.Dropdown(
                            choices=list(MODEL_CONFIG.keys()),
                            value="👁️ Swin Transformer (Vision ViT)",
                            label="Engine to test",
                        )
                        rob_img_input = gr.Image(type="pil", show_label=False, label="Specimen")
                        sigma_slider = gr.Slider(0.0, 0.6, value=0.2, step=0.02, label="Noise level (σ)")
                        noisy_preview = gr.Image(label="Noisy preview", interactive=False)
                        run_rob_btn = gr.Button("🌪️ Run Stress Test", variant="primary")
                        rob_verdict = gr.Markdown("Upload an image, choose a noise level, then run the test.", elem_classes="robustness-verdict")

                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🟢 Clean image", elem_classes="section-label")
                            clean_out = gr.HTML(value=_idle_card("Clean-image result"), elem_classes="glass-panel")
                        with gr.Column():
                            gr.Markdown("### 🌪️ Noisy image", elem_classes="section-label")
                            noisy_out = gr.HTML(value=_idle_card("Noisy-image result"), elem_classes="glass-panel")

            def _rob_loading():
                return _loading_card("Adding noise"), _loading_card("Running inference"), None, "Running stress test…"

            run_rob_btn.click(fn=_rob_loading, inputs=None, outputs=[clean_out, noisy_out, noisy_preview, rob_verdict]).then(
                fn=run_robustness_test,
                inputs=[rob_img_input, rob_model_dd, sigma_slider],
                outputs=[clean_out, noisy_out, noisy_preview, rob_verdict],
            )

        # ---------------- GRAD-CAM TAB ----------------
        with gr.Tab("🔥 Grad-CAM", id=2):
            gr.Markdown("### 🔥 See where the model is actually looking", elem_classes="section-label")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="control-card"):
                        cam_model_dd = gr.Dropdown(
                            choices=list(MODEL_CONFIG.keys()),
                            value="🏆 ConvNeXt-Tiny (Champion)",
                            label="Engine (not available for Swin Transformer)",
                        )
                        cam_img_input = gr.Image(type="pil", show_label=False, label="Specimen")
                        run_cam_btn = gr.Button("🔥 Run Grad-CAM", variant="primary")
                        cam_caption = gr.Markdown(
                            "Upload a leaf image, pick a CNN engine, and run Grad-CAM to see a heatmap "
                            "of the exact regions that drove its diagnosis.",
                            elem_classes="robustness-verdict",
                        )
                with gr.Column(scale=2):
                    gr.Markdown("### 🌡️ Attention heatmap", elem_classes="section-label")
                    cam_output = gr.Image(label=None, show_label=False, elem_classes="glass-panel")

            def _cam_loading():
                return None, "Running Grad-CAM…"

            run_cam_btn.click(fn=_cam_loading, inputs=None, outputs=[cam_output, cam_caption]).then(
                fn=run_gradcam, inputs=[cam_img_input, cam_model_dd], outputs=[cam_output, cam_caption]
            )

        # ---------------- DATASET GALLERY TAB ----------------
        with gr.Tab("🗂 Dataset Gallery", id=3):
            gr.Markdown("### 🌱 Browse the PlantVillage training set live", elem_classes="section-label")
            with gr.Row():
                species_dd = gr.Dropdown(
                    choices=["All species"] + SPECIES, value="All species",
                    label="Filter by species", scale=2,
                )
                class_dd = gr.Dropdown(
                    choices=["All classes"] + CLASS_LABELS, value="All classes",
                    label="Filter by exact disease class (auto-narrows to the species above)", scale=3,
                )
                load_gallery_btn = gr.Button("🔄 Load live samples", variant="primary", scale=1)

            species_dd.change(fn=classes_for_species, inputs=[species_dd], outputs=[class_dd])

            gallery_status = gr.Markdown(
                "Click **Load live samples** to stream real images from the dataset used to train every model above. "
                "Pick a species for a mixed view, or an exact class to see all ~36 images in that folder.",
                elem_classes="gallery-note",
            )
            gallery = gr.Gallery(label=None, show_label=False, columns=4, height=520, object_fit="cover", elem_classes="gallery")
            send_to_diagnose_btn = gr.Button("➡️ Send to Diagnose + Grad-CAM + Robustness Lab", variant="secondary")
            gallery_selected_img = gr.State(None)

            load_gallery_btn.click(fn=load_gallery, inputs=[species_dd, class_dd], outputs=[gallery, gallery_status])

            def _pick(evt: gr.SelectData):
                val = evt.value
                if isinstance(val, dict):
                    img = val.get("image", val)
                    if isinstance(img, dict):
                        return img.get("path") or img.get("url")
                    return img
                if isinstance(val, (list, tuple)) and val:
                    first = val[0]
                    if isinstance(first, dict):
                        return first.get("path") or first.get("url")
                    return first
                return val

            def _send_selected_everywhere(img):
                """Sends the picked gallery image to the Diagnose tab, the
                Grad-CAM tab, AND the Robustness Lab tab in one click, confirms
                it with a toast, and auto-switches the view to Diagnose so the
                user isn't left guessing whether anything happened."""
                if img is None:
                    gr.Warning("Click a thumbnail in the gallery first, then hit Send.")
                    return gr.update(), gr.update(), gr.update(), gr.Tabs()
                gr.Info("✅ Sent to Diagnose, Grad-CAM, and Robustness Lab — switching tab…")
                return img, img, img, gr.Tabs(selected=0)

            gallery.select(fn=_pick, inputs=None, outputs=gallery_selected_img)
            send_to_diagnose_btn.click(
                fn=_send_selected_everywhere,
                inputs=[gallery_selected_img],
                outputs=[img_input, rob_img_input, cam_img_input, main_tabs],
            )

            gr.HTML(build_species_grid_html())

        # ---------------- LEADERBOARD TAB ----------------
        with gr.Tab("🏆 Leaderboard", id=4):
            gr.HTML(build_leaderboard_html())

        # ---------------- ABOUT TAB ----------------
        with gr.Tab("📚 About", id=5):
            gr.HTML(ABOUT_HTML)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)