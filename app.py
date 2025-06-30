# app.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Define the model class (same architecture as training)
class ASLClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

@st.cache_resource
def load_model(path):
    checkpoint = torch.load(path, map_location='cpu')
    classes = checkpoint['classes']
    model = ASLClassifier(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, classes

model, class_labels = load_model("asl_model.pth")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Streamlit Interface
st.title("ASL Alphabet Classifier ✋")
choice = st.radio("Choose input method", ["Upload Image", "Webcam (optional)"])

if choice == "Upload Image":
    img_file = st.file_uploader("Upload an image of a sign:")
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs.data, 1)
            prediction = class_labels[pred.item()]
            st.success(f"Predicted Sign: **{prediction}**")
