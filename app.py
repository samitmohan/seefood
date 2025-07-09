import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path

def load_model(model_path: str):
    try:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image):
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def get_prediction(model, image_tensor):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            return predicted.item(), confidence.item()
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def main():
    st.set_page_config(
        page_title="🌭 SeeFood - Hotdog Detector",
        page_icon="🌭",
        layout="centered"
    )
    
    st.title("🌭 SeeFood")
    st.write("**Is it a hotdog, or not a hotdog?**")
    
    model_path = Path('hotdog_model.pth')
    if not model_path.exists():
        st.error("Model file not found. Please run `python model.py` to train the model first.")
        return
    
    model = load_model(str(model_path))
    if model is None:
        return
    
    uploaded_file = st.file_uploader(
        "Upload an image to find out...",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            image_tensor = process_image(image)
            if image_tensor is not None:
                with st.spinner("Thinking..."):
                    predicted, confidence = get_prediction(model, image_tensor)
                
                if predicted is not None:
                    is_hotdog = (predicted == 0)
                    label = "Hotdog" if is_hotdog else "Not Hotdog"
                    icon = "🌭" if is_hotdog else "❌"
                    
                    st.markdown(f'<h2 style="text-align: center;">{icon} {label}</h2>', unsafe_allow_html=True)
                    st.markdown(f'<p style="text-align: center;">Confidence: {confidence:.2%}</p>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()