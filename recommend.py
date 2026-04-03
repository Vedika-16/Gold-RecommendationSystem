import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def image_to_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_t).squeeze().numpy()
    return embedding / np.linalg.norm(embedding)

# Load dataset
dataset_path = "dataset"
image_paths = []
categories = []
embeddings = []

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    for file in os.listdir(category_path):
        if file.endswith(('jpg','png','jpeg')):
            path = os.path.join(category_path, file)
            image_paths.append(path)
            categories.append(category)
            embeddings.append(image_to_embedding(path))

embeddings = np.array(embeddings)

# Query example
query_image = "dataset/rings/ring1.jpg"
query_emb = image_to_embedding(query_image)

similarities = cosine_similarity([query_emb], embeddings)[0]
top_indices = similarities.argsort()[-5:][::-1]

print("Top recommendations:")
for idx in top_indices:
    print(image_paths[idx], "| Category:", categories[idx], "| Score:", similarities[idx])
