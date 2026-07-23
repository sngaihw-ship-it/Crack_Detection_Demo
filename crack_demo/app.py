import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from huggingface_hub import hf_hub_download

# ---------- Get the directory where this script lives ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Model definition (UNet16 with VGG16) ----------
class Interpolate(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.conv(x))

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super().__init__()
        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )
    def forward(self, x):
        return self.block(x)

class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = models.vgg16(pretrained=pretrained).features
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0], self.relu, self.encoder[2], self.relu)
        self.conv2 = nn.Sequential(self.encoder[5], self.relu, self.encoder[7], self.relu)
        self.conv3 = nn.Sequential(self.encoder[10], self.relu, self.encoder[12], self.relu, self.encoder[14], self.relu)
        self.conv4 = nn.Sequential(self.encoder[17], self.relu, self.encoder[19], self.relu, self.encoder[21], self.relu)
        self.conv5 = nn.Sequential(self.encoder[24], self.relu, self.encoder[26], self.relu, self.encoder[28], self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)

# ---------- Model loader (Hugging Face Hub) ----------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Sansweeper/Crack_Detection",
        filename="model_unet_vgg_16_best.pt"
    )
    model = UNet16(pretrained=False)
    state = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

# ---------- Image Enhancement ----------
def enhance_image(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

# ---------- Preprocessing ----------
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ---------- Inference ----------
def process_image(img_rgb, enhance=False, threshold=0.2):
    h, w = img_rgb.shape[:2]
    if enhance:
        img_rgb = enhance_image(img_rgb)
    img_resized = cv2.resize(img_rgb, (448, 448), interpolation=cv2.INTER_AREA)
    inp = transform(Image.fromarray(img_resized)).unsqueeze(0)
    with torch.no_grad():
        out = model(inp)
        prob = torch.sigmoid(out).squeeze().cpu().numpy()
    prob_resized = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

    prob_uint8 = (prob_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    mask = (prob_resized > threshold).astype(np.uint8) * 255

    overlay = img_rgb.copy()
    red = np.zeros_like(overlay)
    red[:,:,0] = mask
    overlay = cv2.addWeighted(overlay, 0.6, red, 0.4, 0)

    dark = np.zeros_like(overlay)
    dark[:,:,0] = mask
    dark[:,:,1] = mask
    dark[:,:,2] = mask

    return {
        "original": img_rgb,
        "enhanced": img_rgb if not enhance else enhance_image(img_rgb),
        "heatmap": heatmap,
        "overlay": overlay,
        "dark": dark,
        "mask": mask
    }

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Crack Segmentation Pro", layout="wide")
st.title("Crack Detection - Samuel Ngai")

model = load_model()

st.sidebar.title("Controls")

# ---------- Robust sample loading ----------
# Look for 'samples' folder in the same directory as this script
sample_dir = os.path.join(SCRIPT_DIR, "samples")
if not os.path.exists(sample_dir):
    # Fallback: try one level up (if samples is at repo root)
    sample_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "samples")

samples = {}
if os.path.exists(sample_dir):
    sample_files = [f for f in os.listdir(sample_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if sample_files:
        samples = {os.path.splitext(f)[0]: os.path.join(sample_dir, f) for f in sample_files}
    else:
        st.sidebar.warning("No image files found in 'samples/' folder.")
else:
    st.sidebar.warning("No 'samples/' folder found. Please upload an image.")

# Dropdown if samples exist
if samples:
    choice = st.sidebar.selectbox("Pick a sample", list(samples.keys()))
else:
    choice = None

uploaded = st.sidebar.file_uploader("Or upload your own", type=["jpg","jpeg","png"])
enhance = st.sidebar.checkbox("Enhance Contrast & Sharpen Edges", value=False)
threshold = st.sidebar.slider("Threshold", 0.05, 0.8, 0.2, 0.05)

# Load image
if uploaded is not None:
    img = np.array(Image.open(uploaded).convert("RGB"))
elif choice is not None and samples:
    img = np.array(Image.open(samples[choice]).convert("RGB"))
else:
    st.warning("Please upload an image or add sample images to the 'samples/' folder.")
    st.stop()

if st.sidebar.button("SEGMENT!"):
    with st.spinner("Running inference..."):
        result = process_image(img, enhance=enhance, threshold=threshold)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(result["original"], caption="Original", use_container_width=True)
        if enhance:
            st.image(result["enhanced"], caption="Enhanced (CLAHE + Unsharp)", use_container_width=True)
    with col2:
        st.image(result["heatmap"], caption="Probability Heatmap", use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.image(result["overlay"], caption="Overlay (Cracks highlighted)", use_container_width=True)
    with col4:
        st.image(result["dark"], caption="Binary Mask", use_container_width=True)
