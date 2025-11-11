import streamlit as st
import torch, numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
from src.utils import CITYSCAPES_TRAINID_COLORS

st.set_page_config(page_title="Datu kopa Cityscapes - Semantiskā segmentācija", layout="wide")
st.title("Cityscapes - segmentācija no 19 klasēm, izmantojot Unet tīklu")

weights = st.text_input("Svaru ceļš", value="runs/ckpt.pt")
encoder = st.text_input("Enkoders", value="resnet34")
arch = st.text_input("Arhitektūra", value="Unet")
img_size_h = st.number_input("Att. izmērs, augstums (h)", value=96, min_value=28, max_value=1024, step=1)
img_size_w = st.number_input("Att. izmērs, platums (w)", value=256, min_value=28, max_value=1024, step=1)

@st.cache_resource
def load_model(weights, encoder, arch, img_size_h, img_size_w, num_classes=19):
    model_cls = getattr(smp, arch)
    m = model_cls(encoder_name=encoder, encoder_weights=None, classes=num_classes, activation=None)
    try:
        sd = torch.load(weights, map_location="cpu")
        m.load_state_dict(sd, strict=False)
        st.success("Modelis ielādēts.")
    except Exception as e:
        st.warning(f"Nesanāca ielādēt svarus: {e}. Izmantojam gadījuma svarus.")
    m.eval()
    tfm = transforms.Compose([transforms.Resize((img_size_h,img_size_w)), transforms.ToTensor()])
    return m, tfm

def colorize(mask):
    # mask: HxW trainId values [0..18], others -> 255 ignored
    h,w = mask.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    for tid, color in enumerate(CITYSCAPES_TRAINID_COLORS):
        out[mask==tid] = color
    return out

uploaded = st.file_uploader("Augšupielādējiet ielas attēlu (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_container_width=True)
    model, tfm = load_model(weights, encoder, arch, img_size_h,img_size_w)
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1)[0].numpy()
    st.image(colorize(pred), caption="Prognozētā segmentācija", use_container_width=True)
