import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import os

CLASS_NAMES = ['baluchari', 'chanderi', 'kanjeevaram', 'kosa',
               'kota_doria', 'muga', 'paithani', 'patola',
               'pochampally', 'uppada']
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE    = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
REPO_ID     = 'SiddhantDCT/silk'

REGION_GUIDANCE = {
    'chanderi'   : 'Photograph the small gold buti motifs on the fabric body',
    'kanjeevaram': 'Photograph the wide border of the saree',
    'kosa'       : 'Photograph the fabric body texture up close (6-8 cm)',
    'paithani'   : 'Photograph the peacock motif in the pallu',
    'patola'     : 'Photograph the geometric ikat pattern on the body',
    'uppada'     : 'Photograph the floating woven motifs on the body',
    'pochampally': 'Photograph the geometric diamond ikat pattern',
    'muga'       : 'Photograph the fabric body to show the golden sheen',
    'baluchari'  : 'Photograph the pallu showing the narrative figure motifs',
    'kota_doria' : 'Photograph the fabric against light to show khat checks',
}

@st.cache_resource
def load_models():
    st.write("Loading models from Hugging Face...")

    resnet_path = hf_hub_download(
        repo_id=REPO_ID, filename='resnet50_best.pth')
    mobile_path = hf_hub_download(
        repo_id=REPO_ID, filename='mobilenet_v2_best.pth')

    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(resnet.fc.in_features, NUM_CLASSES))
    resnet.load_state_dict(
        torch.load(resnet_path, map_location='cpu'))
    resnet.eval()

    mobile = models.mobilenet_v2(weights=None)
    mobile.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(mobile.classifier[1].in_features, NUM_CLASSES))
    mobile.load_state_dict(
        torch.load(mobile_path, map_location='cpu'))
    mobile.eval()

    return resnet, mobile


def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor)[0], dim=0).numpy()
    return probs


# ── PAGE ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Silk Saree Classifier",
    layout="wide"
)

st.title("Indian Silk Saree Classification")
st.caption(
    "10 GI-Tagged Varieties  |  "
    "ResNet-50: 95.72%  |  MobileNetV2: 95.39%  |  "
    "NIT Silchar Research Project"
)
st.divider()

resnet_model, mobile_model = load_models()

# ── SETTINGS ───────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    model_choice = st.radio(
        "Model",
        ["ResNet-50 (95.72% accuracy)",
         "MobileNetV2 (95.39% accuracy, lightweight)"],
        horizontal=False
    )
with col2:
    mode = st.radio(
        "Mode",
        ["Single image",
         "Multi-image ensemble (5-10 images, Prof Roy method)"],
        horizontal=False
    )

active_model = (resnet_model
                if "ResNet" in model_choice
                else mobile_model)
active_label = ("ResNet-50" if "ResNet" in model_choice
                else "MobileNetV2")

st.divider()

# ── UPLOAD ─────────────────────────────────────────────────────
if "Single" in mode:
    files = st.file_uploader(
        "Upload one saree photograph",
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=False
    )
    if files:
        files = [files]
else:
    st.info(
        "Upload 5-10 photographs of the same saree from "
        "different angles and regions: full view, border "
        "close-up, pallu, body texture, zari detail. "
        "The system will average predictions across all images."
    )
    files = st.file_uploader(
        "Upload 5-10 saree photographs",
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=True
    )

# ── PREDICT ────────────────────────────────────────────────────
if files and len(files) > 0:
    images = [Image.open(f).convert('RGB') for f in files]

    # Run prediction on each image
    all_probs = [predict_image(img, active_model)
                 for img in images]
    avg_probs   = np.mean(all_probs, axis=0)
    pred_idx    = avg_probs.argmax()
    final_class = CLASS_NAMES[pred_idx]
    final_conf  = avg_probs[pred_idx] * 100

    individual = [
        {'pred': CLASS_NAMES[p.argmax()], 'conf': p.max()*100}
        for p in all_probs
    ]
    agreement = sum(
        r['pred'] == final_class for r in individual)

    # Show uploaded images
    n    = len(images)
    cols = st.columns(min(n, 5))
    for i, (img, col) in enumerate(zip(images[:5], cols)):
        with col:
            ind = individual[i]
            match = "OK" if ind['pred'] == final_class else "--"
            st.image(img, use_column_width=True,
                     caption=(f"Image {i+1} | "
                              f"[{match}] {ind['pred']} "
                              f"({ind['conf']:.0f}%)"))
    if n > 5:
        cols2 = st.columns(min(n-5, 5))
        for i, (img, col) in enumerate(
                zip(images[5:], cols2)):
            with col:
                ind = individual[i+5]
                match = ("OK" if ind['pred'] == final_class
                         else "--")
                st.image(img, use_column_width=True,
                         caption=(f"Image {i+6} | "
                                  f"[{match}] {ind['pred']} "
                                  f"({ind['conf']:.0f}%)"))

    st.divider()

    # Result
    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        st.subheader("Result")

        if final_conf >= 90:
            st.success(
                f"{final_class.upper()}  |  "
                f"{final_conf:.1f}% confidence")
        elif final_conf >= 40:
            st.warning(
                f"{final_class.upper()}  |  "
                f"{final_conf:.1f}% confidence")
            st.info(REGION_GUIDANCE[final_class])
        else:
            st.error(
                f"Not recognized  |  {final_conf:.1f}%")
            st.write(
                "The uploaded image may not match any of "
                "the 10 trained silk saree categories.")

        if n > 1:
            st.metric(
                "Image Agreement",
                f"{agreement} / {n}",
                help="Number of images that individually "
                     "predicted the same final class")

        st.caption(f"Model: {active_label}")

    with col_r2:
        st.subheader("Class Probabilities")
        sorted_idx = np.argsort(avg_probs)[::-1]
        for idx in sorted_idx:
            cls  = CLASS_NAMES[idx]
            prob = float(avg_probs[idx] * 100)
            if idx == pred_idx:
                st.markdown(f"**{cls}: {prob:.1f}%**")
            else:
                st.write(f"{cls}: {prob:.1f}%")
            st.progress(int(prob))

    st.divider()
    st.caption(
        "Two-stage classification: if confidence is below 90%, "
        "follow the guidance above to photograph the most "
        "distinctive feature of the predicted saree type. "
        "Multi-image ensemble averages predictions across all "
        "uploaded region photographs for improved accuracy."
    )