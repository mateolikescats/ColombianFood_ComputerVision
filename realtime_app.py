import sys
import json
import joblib
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms, models
import torch.nn as nn
import streamlit as st

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("Processed")

@st.cache_resource
def load_cnn_model():
    cnn_path = MODELS_DIR / "best_cnn_model.pth"
    if not cnn_path.exists():
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cnn_path, map_location=device)
    class_names = checkpoint['class_names']
    model_name = checkpoint.get('model_name', 'mobilenet_v3_small')
    
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small()
        if 'best_params' in checkpoint:
            dropout_rate = checkpoint['best_params'].get('dropout_rate', 0.2)
            model.classifier[2] = nn.Dropout(p=dropout_rate)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, len(class_names))
    elif model_name == "resnet18":
        model = models.resnet18()
        in_features = model.fc.in_features
        if "fc.1.weight" in checkpoint['state_dict']:
            dropout_rate = checkpoint.get('best_params', {}).get('dropout_rate', 0.2)
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, len(class_names))
            )
        else:
            model.fc = nn.Linear(in_features, len(class_names))
    else:
        return None, None
        
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model, class_names

@st.cache_resource
def load_ml_model():
    ml_path = MODELS_DIR / "best_ml_model.pkl"
    if not ml_path.exists():
        return None, None, None
    artifact = joblib.load(ml_path)
    return artifact['model'], artifact['scaler'], artifact['categories']

def predict_cnn(image_pil, model, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
    top_idx = np.argmax(probabilities)
    return class_names[top_idx], probabilities[top_idx], probabilities

def run_streamlit_app():
    st.set_page_config(page_title="Clasificador de Comida Colombiana", page_icon="🇨🇴", layout="centered")
    
    st.title("🇨🇴 Clasificador de Comida Colombiana")
    st.markdown("Reconocimiento en tiempo real de 15 platos y snacks típicos colombianos mediante Visión Artificial.")
    
    cnn_model, class_names = load_cnn_model()
    
    if cnn_model is None:
        st.warning("⚠️ No se encontró el modelo entrenado. Ejecuta primero `preprocess_dataset.py` y `train_cnn.py`.")
        return

    st.sidebar.header("⚙️ Configuración")
    input_source = st.sidebar.radio("Selecciona la fuente de entrada:", ["📷 Cámara web en directo", "📁 Subir imagen"])
    
    image_to_process = None
    
    if input_source == "📷 Cámara web en directo":
        img_file_buffer = st.camera_input("Toma una fotografía de la comida colombiana:")
        if img_file_buffer is not None:
            image_to_process = Image.open(img_file_buffer)
    else:
        uploaded_file = st.file_uploader("Elige una imagen JPG, PNG o HEIC...", type=["jpg", "jpeg", "png", "heic"])
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            st.image(image_to_process, caption="Imagen cargada", use_column_width=True)

    if image_to_process is not None:
        pred_label, confidence, probs = predict_cnn(image_to_process, cnn_model, class_names)
        
        st.success(f"### 🍽️ Predicción: **{pred_label}**")
        st.metric(label="Nivel de Confianza", value=f"{confidence * 100:.2f}%")
        
        # Show top 3 predictions
        st.subheader("Top 3 Categorías Probables:")
        top3_indices = np.argsort(probs)[::-1][:3]
        for idx in top3_indices:
            st.write(f"- **{class_names[idx]}**: {probs[idx]*100:.2f}%")
            st.progress(float(probs[idx]))

def run_opencv_webcam():
    cnn_model, class_names = load_cnn_model()
    if cnn_model is None:
        print("Model checkpoint not found. Train CNN first!")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to exit OpenCV live stream...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert OpenCV BGR frame to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        label, conf, _ = predict_cnn(pil_img, cnn_model, class_names)

        # Draw prediction overlay
        text = f"{label}: {conf * 100:.1f}%"
        cv2.rectangle(frame, (10, 10), (450, 60), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Colombian Food Classifier (Real-Time)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--opencv":
        run_opencv_webcam()
    else:
        run_streamlit_app()
