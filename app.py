import os, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import altair as alt
import cv2
import pydicom
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------
# Paths
# -----
ROOT          = Path("/Volumes/DUAL DRIVE/Azmina_MammoMind_FYP")
PROCESSED     = ROOT / "processed"
MODELS_DIR    = PROCESSED / "models"
LOGS_DIR      = PROCESSED / "logs"
METRICS_DIR   = PROCESSED / "metrics"
PLOTS_DIR     = PROCESSED / "plots"
EVAL_DIR      = PROCESSED / "eval_attention"
GRADCAM_DIR   = EVAL_DIR / "gradcam"
EXPORTS_DIR   = PROCESSED / "exports"

ATTN_CKPT     = MODELS_DIR / "attention_cnn.best.h5"
INPUT_SIZE    = (64, 64)
LAST_CONV_NAME_DEFAULT = "b4_conv"

# logo
LOGO_PATH     = ROOT / "assets" / "logo.png"

# -----------------------------
# Streamlit config + CSS theme
# -----------------------------
st.set_page_config(page_title="MammoMind: Breast Cancer Detection", layout="wide")

st.markdown(
    """
    <style>
      /* Page width & background */
      .main .block-container { max-width: 1280px; padding-top: 1.2rem; }
      body {
        background: linear-gradient(135deg,#f7fafc 0%, #eef2ff 100%);
      }
      /* Header */
      .header-wrap { display:flex; align-items:center; gap:16px; margin-bottom: 8px; }
      .header-wrap img { height: 128px; filter: drop-shadow(0 4px 8px rgba(0,0,0,.12)); }
      .header-wrap img { height: 128px; filter: drop-shadow(0 4px 8px rgba(0,0,0,.12)); }
      .header-title h1 { margin: 0; font-size: 26px; color:#0f172a; }
      .header-title p  { margin: 2px 0 0; color:#475569; }

      /* Tabs */
      .stTabs [data-baseweb="tab"]{
        padding: 10px 18px; border-radius: 12px; background: rgba(255,255,255,.55);
        box-shadow: 0 2px 10px rgba(0,0,0,.06); margin-right: 8px;
        backdrop-filter: blur(6px);
      }
      .stTabs [aria-selected="true"]{
        background: rgba(255,255,255,.9); font-weight: 700; color:#111827;
      }

      /* Glassy widgets */
      .stFileUploader, .stSlider, .stSelectbox, .stMultiSelect, .stTextInput {
        background: rgba(255,255,255,.6)!important; border-radius: 14px!important;
        border: 1px solid rgba(15,23,42,.08)!important; padding: 6px 10px!important;
        box-shadow: 0 6px 18px rgba(15,23,42,.06)!important; backdrop-filter: blur(6px);
      }
      .stButton>button {
        background: rgba(255,255,255,.7); color:#0f172a; border: 1px solid rgba(15,23,42,.08);
        border-radius: 12px; padding: 8px 14px; font-weight: 700;
        box-shadow: 0 10px 20px rgba(15,23,42,.08); backdrop-filter: blur(6px);
      }
      .stButton>button:hover { transform: translateY(-1px); border-color:#3b82f6; }

      /* Section labels */
      h2, h3 { color:#0f172a; }
      .pill { padding:10px 14px;border-radius:12px;font-weight:700;display:inline-block; }
      .pill.green { background:#e9f7ef; color:#0a7d2c; }
      .pill.red   { background:#ffe5e7; color:#b00020; }

      /* Image cards (gallery) */
      .img-card { transition: transform .08s ease, box-shadow .08s ease; }
      .img-card:hover { transform: translateY(-3px); box-shadow: 0 12px 24px rgba(15,23,42,.12); }
      .cap { color:#475569; font-size:.88rem; margin-top:.35rem; text-align:center; }
      .muted { color:#64748b; font-size:.9rem; }
      .kbd { background:#0f172a; color:#fff; border-radius:6px; padding:2px 6px; font-size:.8rem; }
      footer { visibility: hidden; } /* hide default footer */
    </style>
    """,
    unsafe_allow_html=True
)


# Image helpers
def ensure_gray_64(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[:2] != INPUT_SIZE:
        img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img

def read_image_any(file_bytes: bytes, filename: str) -> np.ndarray:
    name = filename.lower()
    if name.endswith(".dcm"):
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        if not hasattr(ds, "pixel_array"):
            raise ValueError("DICOM has no pixel data.")
        arr = ds.pixel_array
        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass
        arr = arr.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        return (arr * 255.0).astype(np.uint8)
    else:
        buf = np.frombuffer(file_bytes, np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ValueError("Could not decode PNG/JPG.")
        return im

# Model definition (SE attention)
def se_block(x, reduction=8):
    c = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(c // reduction, 8), activation="relu")(s)
    s = layers.Dense(c, activation="sigmoid")(s)
    s = layers.Reshape((1,1,c))(s)
    return layers.Multiply()([x, s])

def build_attention_model(input_shape=(64,64,1), last_conv_name="b4_conv"):
    inp = layers.Input(shape=input_shape)
    # Block 1
    x = layers.Conv2D(32,3,padding="same",use_bias=False)(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = se_block(x, reduction=8)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.15)(x)
    # Block 2
    x = layers.Conv2D(64,3,padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = se_block(x, reduction=8)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.20)(x)
    # Block 3
    x = layers.Conv2D(128,3,padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = se_block(x, reduction=8)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.25)(x)
    # Head
    x = layers.Conv2D(256,3,padding="same",use_bias=False,name=last_conv_name)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = se_block(x, reduction=8)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="attention_se_cnn")

@st.cache_resource(show_spinner=False)
def load_model_cached():
    if not ATTN_CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ATTN_CKPT}")
    model = build_attention_model(input_shape=(*INPUT_SIZE,1), last_conv_name=LAST_CONV_NAME_DEFAULT)
    model.load_weights(str(ATTN_CKPT))
    return model

# Inference utilities

def predict_prob(model, gray64: np.ndarray) -> float:
    x = gray64[None, ..., None]
    return float(model.predict(x, verbose=0).astype(np.float32)[0,0])

def mc_dropout_prob(model, gray64: np.ndarray, samples: int = 20) -> tuple[float,float]:
    x = gray64[None, ..., None]
    vals = [model(x, training=True).numpy()[0,0] for _ in range(samples)]
    vals = np.asarray(vals, dtype=np.float32)
    return float(vals.mean()), float(vals.std())

def gradcam_u8(model, gray64: np.ndarray, last_conv_name: str | None = None) -> np.ndarray:
    x = gray64[None, ..., None]
    layer_name = last_conv_name or LAST_CONV_NAME_DEFAULT
    try:
        last_conv = model.get_layer(layer_name)
    except ValueError:
        conv_layers = [l for l in model.layers if isinstance(l, layers.Conv2D)]
        if not conv_layers:
            raise RuntimeError("No Conv2D layers for Grad-CAM.")
        last_conv = conv_layers[-1]
    grad_model = keras.Model([model.inputs], [last_conv.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        loss = preds[:, 0]
    grads  = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    cam    = tf.einsum("hwc,c->hw", conv_out[0], pooled)
    cam    = tf.maximum(cam, 0) / (tf.reduce_max(cam)+1e-6)
    cam    = cv2.resize(cam.numpy().astype(np.float32), INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
    return np.clip(cam*255.0, 0, 255).astype(np.uint8)

def overlay_cam(gray64: np.ndarray, cam8: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    base = (gray64*255.0).astype(np.uint8)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    heat = cv2.applyColorMap(cam8, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(base_rgb, 1.0, heat, alpha, 0)

def cam_outline(cam8: np.ndarray, frac: float = 0.15) -> np.ndarray:
    thresh = np.quantile(cam8.flatten(), 1.0 - frac)
    return (cam8 >= thresh).astype(np.uint8)*255

def draw_outline_on_gray(gray64: np.ndarray, mask255: np.ndarray, color=(0,255,0)) -> np.ndarray:
    base = (gray64*255.0).astype(np.uint8)
    rgb  = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    cnts, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(rgb, cnts, -1, color, 2)
    return rgb

# Header logo + title
c_logo, c_title = st.columns([1, 9])
with c_logo:
    if LOGO_PATH.exists():
        st.markdown("<div class='header-wrap'>", unsafe_allow_html=True)
        st.image(str(LOGO_PATH))
        st.markdown("</div>", unsafe_allow_html=True)
with c_title:
    st.markdown(
        "<div class='header-title'>"
        "<h1>MammoMind: Breast Cancer Detection</h1>"
        "<p class='muted'>Patch-based attention CNN • Uncertainty • Grad-CAM</p>"
        "</div>",
        unsafe_allow_html=True
    )

tabs = st.tabs(["Predict", "Metrics", "Grad-CAM Gallery", "About"])

# Predict tab
with tabs[0]:
    lcol, rcol = st.columns([3, 2], gap="large")
    with lcol:
        st.subheader("Predict malignancy from an image")
        st.caption("Upload a **PNG/JPG patch (64×64 preferred)** or a **DICOM** (`.dcm`) image.")
        up = st.file_uploader("Drag & drop or browse (PNG, JPG, JPEG, DCM)", type=["png","jpg","jpeg","dcm"])

    with rcol:
        mc_samples = st.slider("MC Dropout samples", 5, 50, 20, 1,
                               help="More samples → smoother mean/std (but slower).")
        threshold  = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01,
                               help="If MC mean ≥ threshold ⇒ malignant; else benign.")

    st.markdown("---")

    model = load_model_cached()

    if up is None:
        st.info("Upload one image to run prediction, uncertainty, and Grad-CAM.")
    else:
        try:
            raw  = read_image_any(up.read(), up.name)
            gray = ensure_gray_64(raw)
        except Exception as e:
            st.error(f"Failed to read image: {e}")
            st.stop()

        p_single = predict_prob(model, gray)
        mc_mean, mc_std = mc_dropout_prob(model, gray, samples=mc_samples)

        cam8    = gradcam_u8(model, gray, last_conv_name=LAST_CONV_NAME_DEFAULT)
        overlay = overlay_cam(gray, cam8, alpha=0.35)
        mask    = cam_outline(cam8, frac=0.15)
        pred_lbl = "malignant" if mc_mean >= threshold else "benign"
        outline_color_bgr = (217, 4, 41) if pred_lbl=="malignant" else (10, 125, 44)
        outline = draw_outline_on_gray(gray, mask, color=outline_color_bgr)

        c1, c2, c3, c4 = st.columns(4)
        c1.image(gray,    caption="Preprocessed (64×64)", use_container_width=True, clamp=True)
        c2.image(cam8,    caption="Grad-CAM (0–255)",     use_container_width=True, clamp=True)
        c3.image(overlay, caption="Overlay (TURBO colormap)", use_container_width=True)
        c4.image(outline, caption="Top-15% CAM outline",  use_container_width=True)

        st.markdown("### Decision")
        is_malig = (pred_lbl == "malignant")
        pill_cls = "red" if is_malig else "green"
        st.markdown(
            f"<span class='pill {pill_cls}'>Decision: {pred_lbl.upper()}  "
            f"(threshold={threshold:.2f}) • single={p_single:.3f} • MC={mc_mean:.3f}±{mc_std:.3f}</span>",
            unsafe_allow_html=True
        )

# Metrics tab
with tabs[1]:
    st.subheader("Evaluation metrics & plots")
    
    def first_existing(*cands):
        for c in cands:
            if c and Path(c).exists():
                return Path(c)
        return None
    
# --- Training History ---
    st.markdown("#### Training & Validation Curves")
    hist_path = first_existing(LOGS_DIR/"attention_history.csv", LOGS_DIR/"baseline_history.csv")
    if not hist_path:
        for p in sorted(LOGS_DIR.glob("*.csv")):
            try:
                df_ = pd.read_csv(p)
                if "epoch" in df_.columns:
                    hist_path = p; break
            except Exception:
                pass
                
    if hist_path:
        df = pd.read_csv(hist_path)
        curves = [c for c in ["loss","val_loss","accuracy","val_accuracy","auc","val_auc"] if c in df.columns]
        if curves:
            mdf = df[["epoch"]+curves].melt("epoch", var_name="metric", value_name="value")
            chart = alt.Chart(mdf).mark_line(point=True).encode(
                x=alt.X("epoch:Q", title="Epoch"),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("metric:N", legend=alt.Legend(title="")),
                tooltip=["epoch","metric","value"]
            ).properties(height=340)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Found a history CSV but no standard columns to plot.")
    else:
        st.info("No training history CSV found.")

    st.markdown("---")

    # Single JSON metrics - bars (auto-discovery from /metrics or /exports)
    st.markdown("#### Test Metrics & Confusion Matrix")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        export_dirs = sorted(EXPORTS_DIR.glob("fyp_results_*"), reverse=True)
        latest_export = export_dirs[0] if export_dirs else None

        json_candidates = []
        if METRICS_DIR.exists(): json_candidates += sorted(METRICS_DIR.glob("*.json"))
        if latest_export and (latest_export/"metrics").exists():
            json_candidates += sorted((latest_export/"metrics").glob("*.json"))

        picked_json, data = None, None
        for j in json_candidates:
            try:
                cand = json.loads(Path(j).read_text())
                if isinstance(cand, dict) and all(np.isscalar(v) for v in cand.values()):
                    picked_json, data = j, cand; break
            except Exception:
                continue

        if picked_json:
            dfm = pd.DataFrame({"metric": list(data.keys()), "value": list(data.values())})
            bars = alt.Chart(dfm).mark_bar(size=30).encode(
                x=alt.X("metric:N", sort="-y", title=None),
                y=alt.Y("value:Q", title="Score"),
                color=alt.Color("metric:N", legend=None),
                tooltip=["metric","value"]
            ).properties(height=300)
            st.altair_chart(bars, use_container_width=True)
        else:
            st.info("No test-metrics JSON found yet.")

    with col2:
        cm_csv = first_existing(METRICS_DIR/"confusion_matrix.csv",
                                (latest_export/"metrics"/"confusion_matrix.csv") if latest_export else None)
        if cm_csv:
            cm = pd.read_csv(cm_csv, index_col=0)
            cm = cm.reset_index().melt("index", var_name="Predicted", value_name="count").rename(columns={"index":"True"})
            heat = alt.Chart(cm).mark_rect().encode(
                x=alt.X("Predicted:N", title="Predicted"),
                y=alt.Y("True:N", title="True"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["True","Predicted","count"]
            ).properties(height=320)
            text = alt.Chart(cm).mark_text(color="black").encode(x="Predicted:N", y="True:N", text="count:Q")
            st.altair_chart(heat + text, use_container_width=True)
        else:
            st.info("No confusion matrix found.")

# Grad-CAM Gallery tab
with tabs[2]:
    st.subheader("Grad-CAM gallery")
    gallery_dir = GRADCAM_DIR if GRADCAM_DIR.exists() else None
    if not gallery_dir and EXPORTS_DIR.exists():
        dirs = sorted(EXPORTS_DIR.glob("fyp_results_*"), reverse=True)
        if dirs and (dirs[0]/"gradcam").exists():
            gallery_dir = dirs[0]/"gradcam"

    if not gallery_dir:
        st.info("No Grad-CAM directory found yet.")
    else:
        manifest = gallery_dir/"gradcam_manifest.csv"
        meta = None
        if manifest.exists():
            try:
                meta = pd.read_csv(manifest)
            except Exception:
                meta = None

        imgs = sorted(gallery_dir.glob("overlay_*.png"))
        if not imgs:
            st.info("No overlay images found.")
        else:
            ncols = 5
            for i in range(0, len(imgs), ncols):
                cols = st.columns(ncols)
                for j in range(ncols):
                    k = i + j
                    if k < len(imgs):
                        cap = imgs[k].name
                        if meta is not None and {"overlay_png","label","pred_prob"} <= set(meta.columns):
                            row = meta.loc[meta["overlay_png"] == str(imgs[k])]
                            if not row.empty:
                                cap = f'{row["label"].iloc[0]} • p̂={float(row["pred_prob"].iloc[0]):.3f}'
                        with cols[j]:
                            st.image(str(imgs[k]), use_container_width=True, caption=None)
                            st.markdown(f"<div class='cap img-card'>{cap}</div>", unsafe_allow_html=True)

# About tab (project + contact)
with tabs[3]:
    st.subheader("About MammoMind")
    st.markdown(
        """
**MammoMind** is a research prototype exploring **patch-based mammography** on the CBIS-DDSM dataset.  
It combines an **SE attention CNN**, **Monte-Carlo Dropout** (uncertainty), and **Grad-CAM** (explainability) in a
lightweight Streamlit UI.

**What you can do here**
- **Predict**: upload a PNG/JPG patch (64×64) or a DICOM image → see probability, MC mean±std, heatmap, and outline.  
- **Metrics**: view training curves, summary test metrics, and the confusion matrix.  
- **Grad-CAM Gallery**: browse saved overlays.

**Contact**
- Project: *MammoMind* (student research).  
- Email: **MammoMindResearch@outlook.com** - GitHub: **https://github.com/azminajaheer/MammoMind** **Disclaimer**
> This tool is for **education/research only**. It is **not** a medical device and **must not** be used for clinical diagnosis.
        """
    )