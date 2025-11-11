import streamlit as st
import torch, numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
from src.utils import CITYSCAPES_TRAINID_COLORS

st.set_page_config(page_title="Cityscapes — Semantic Segmentation", layout="wide")
st.title("Cityscapes — UNet Segmentation Demo (19 classes)")

weights = st.text_input("Weights path", value="runs/ckpt.pt")
encoder = st.text_input("Encoder", value="resnet34")
arch = st.text_input("Arch (SMP)", value="Unet")
img_size = st.number_input("Image size", value=512, min_value=256, max_value=1024, step=32)

@st.cache_resource
def load_model(weights, encoder, arch, img_size, num_classes=19):
    model_cls = getattr(smp, arch)
    m = model_cls(encoder_name=encoder, encoder_weights=None, classes=num_classes, activation=None)
    try:
        sd = torch.load(weights, map_location="cpu")
        m.load_state_dict(sd, strict=False)
        st.success("Weights loaded.")
    except Exception as e:
        st.warning(f"Could not load weights: {e}. Using random init.")
    m.eval()
    tfm = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
    return m, tfm

def colorize(mask):
    # mask: HxW trainId values [0..18], others -> 255 ignored
    h,w = mask.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    for tid, color in enumerate(CITYSCAPES_TRAINID_COLORS):
        out[mask==tid] = color
    return out

uploaded = st.file_uploader("Upload a street scene image (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_column_width=True)
    model, tfm = load_model(weights, encoder, arch, img_size)
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1)[0].numpy()
    st.image(colorize(pred), caption="Predicted segmentation", use_column_width=True)
