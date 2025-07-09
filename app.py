import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('hotdog_model.pth', map_location='cpu'))
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("🌭 Hotdog / Not Hotdog Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        label = "🌭 Hotdog" if predicted.item() == 0 else "🚫 Not Hotdog"
        st.markdown(f"## Prediction: {label}")