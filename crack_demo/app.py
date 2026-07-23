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

# ---------- Model loader (local path - no gdown needed) ----------
@st.cache_resource
def load_model():
    # Use your existing local weights file
    model_path = r"C:\Users\sngai\Downloads\crack-segmentation-main\crack-segmentation-main\models\model_unet_vgg_16_best.pt"
    
    # If you want to use the Google Drive download instead, uncomment below and comment out the above.
    # if not os.path.exists("model_unet_vgg_16_best.pt"):
    #     import gdown
    #     gdown.download("https://drive.google.com/uc?id=1wA2eAsyFZArG3Zc9OaKvnBuxSAPyDl08", "model_unet_vgg_16_best.pt", quiet=False)
    # model_path = "model_unet_vgg_16_best.pt"

    model = UNet16(pretrained=False)
    state = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

# ---------- Image Enhancement: Contrast + Edge Sharpening ----------
def enhance_image(img_rgb):
    """
    Apply CLAHE (contrast enhancement) and Unsharp Mask (edge sharpening).
    """
    # Convert to LAB for CLAHE (works better on L channel)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Unsharp Mask (edge sharpening)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])  # strength 9
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
def process_image(img_rgb, enhance=False):
    h, w = img_rgb.shape[:2]
    
    # Apply enhancement if requested
    if enhance:
        img_rgb = enhance_image(img_rgb)
    
    # Resize to 448x448
    img_resized = cv2.resize(img_rgb, (448, 448), interpolation=cv2.INTER_AREA)
    inp = transform(Image.fromarray(img_resized)).unsqueeze(0)   # [1,3,448,448]
    
    with torch.no_grad():
        out = model(inp)                                        # [1,1,448,448]
        prob = torch.sigmoid(out).squeeze().cpu().numpy()       # [448,448]  → HEATMAP
    
    # Resize probability back to original dimensions
    prob_resized = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # --- Output 1: Probability Heatmap (color gradient) ---
    # Convert to 0-255 uint8 and apply JET colormap
    prob_uint8 = (prob_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)   # BGR output
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # --- Output 2: Binary Mask (thresholded) ---
    threshold = 0.2
    mask = (prob_resized > threshold).astype(np.uint8) * 255
    
    # --- Output 3: Overlay (red mask on original) ---
    overlay = img_rgb.copy()
    red = np.zeros_like(overlay)
    red[:,:,0] = mask
    overlay = cv2.addWeighted(overlay, 0.6, red, 0.4, 0)
    
    # --- Output 4: Dark background with white cracks ---
    dark = np.zeros_like(overlay)
    dark[:,:,0] = mask
    dark[:,:,1] = mask
    dark[:,:,2] = mask

    return {
        "original": img_rgb,
        "enhanced": img_rgb if not enhance else enhance_image(img_rgb), # show enhanced version if used
        "heatmap": heatmap,
        "overlay": overlay,
        "dark": dark,
        "mask": mask  # optional, for debugging
    }

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Crack Segmentation Pro", layout="wide")
st.title("Crack Detection - Samuel Ngai")

# Load model once
model = load_model()

# Sidebar
st.sidebar.title("Controls")
sample_dir = "samples"
samples = {
    "Asphalt": os.path.join(sample_dir, "Asphalt_deterioration.png"),
    "Bridge":  os.path.join(sample_dir, "Bridge-5.png"),
    "Soil":    os.path.join(sample_dir, "Cracking-Soil.png"),
}
choice = st.sidebar.selectbox("Pick a sample", list(samples.keys()))
uploaded = st.sidebar.file_uploader("Or upload your own", type=["jpg","jpeg","png"])
enhance = st.sidebar.checkbox("Enhance Contrast & Sharpen Edges", value=False)

# Load image
if uploaded is not None:
    img = np.array(Image.open(uploaded).convert("RGB"))
else:
    # Check if sample exists, if not, use a dummy or show error
    sample_path = samples[choice]
    if os.path.exists(sample_path):
        img = np.array(Image.open(sample_path).convert("RGB"))
    else:
        st.error(f"Sample file not found: {sample_path}. Please upload an image.")
        st.stop()

if st.sidebar.button("SEGMENT!"):
    with st.spinner("Running inference..."):
        result = process_image(img, enhance=enhance)
    
    # Display results in 3 rows? Let's do a grid.
    col1, col2 = st.columns(2)
    with col1:
        st.image(result["original"], caption="Original", use_container_width=True)
        if enhance:
            st.image(result["enhanced"], caption="Enhanced (CLAHE + Unsharp)", use_container_width=True)
    with col2:
        st.image(result["heatmap"], caption="Probability Heatmap (Gradient)", use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.image(result["overlay"], caption="Overlay (Cracks highlighted)", use_container_width=True)
    with col4:
        st.image(result["dark"], caption="Binary Mask (White cracks)", use_container_width=True)