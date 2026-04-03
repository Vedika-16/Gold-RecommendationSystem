import os
import torch
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms

# -----------------------------
# Load Model (cached so it loads once)
# -----------------------------
@st.cache_resource
def load_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image preprocessing
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def image_to_embedding(image):
    img = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img).squeeze().numpy()
    return embedding / np.linalg.norm(embedding)

# -----------------------------
# Load dataset embeddings
# -----------------------------
@st.cache_resource
def load_dataset_embeddings():
    dataset_path = "dataset"
    image_paths = []
    embeddings = []

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        for file in os.listdir(category_path):
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                path = os.path.join(category_path, file)
                try:
                    img = Image.open(path).convert("RGB")
                    emb = image_to_embedding(img)
                    image_paths.append(path)
                    embeddings.append(emb)
                except:
                    continue

    return image_paths, np.array(embeddings)

image_paths, embeddings = load_dataset_embeddings()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💎 Gold Jewelry Recommendation System")

uploaded_file = st.file_uploader("Upload a jewelry image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Uploaded Image", width=250)

    query_emb = image_to_embedding(query_image)

    similarities = cosine_similarity([query_emb], embeddings)[0]
    top_indices = similarities.argsort()[::-1][:5]

    st.subheader("🔎 Similar Jewelry Recommendations")

    cols = st.columns(5)

    for i, idx in enumerate(top_indices):
        with cols[i]:
            img = Image.open(image_paths[idx])
            st.image(img, use_container_width=True)
