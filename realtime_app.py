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
    
    st.sidebar.markdown("""
    ### 📖 Instrucciones de Uso:
    1. **Selecciona la Fuente**: Usa la cámara web en directo o sube una imagen de tu plato.
    2. **Encuadre**: Intenta centrar el alimento en la imagen y mantener una distancia moderada (30-50 cm).
    3. **Autodetección (Bounding Box)**: El sistema aislará automáticamente el contorno del alimento de manera cromática para centrar la clasificación y descartar ruido de fondo.
    
    ### ⚠️ Consideraciones al Usar:
    - **Contraste de Fondo**: El algoritmo de bounding box busca objetos saturados. Funciona de manera óptima sobre platos blancos o grises y fondos neutros.
    - **Confusión de Color**: Platos decorados con dibujos o mesas de madera con tonos dorados/marrones pueden confundir la segmentación y alterar el cuadro delimitador.
    - **Iluminación**: Evita sombras demasiado marcadas o destellos directos que alteren la saturación real de la comida.
    """)
    
    image_to_process = None
    
    if input_source == "📷 Cámara web en directo":
        img_file_buffer = st.camera_input("Toma una fotografía de la comida colombiana:")
        if img_file_buffer is not None:
            image_to_process = Image.open(img_file_buffer)
    else:
        uploaded_file = st.file_uploader("Elige una imagen JPG, PNG o HEIC...", type=["jpg", "jpeg", "png", "heic"])
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)

    if image_to_process is not None:
        # Detectar Bounding Box cromático en Streamlit
        img_np = np.array(image_to_process)
        frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Segmentación HSV Otsu
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        _, thresh = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        box_coords = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 2000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                box_coords = (x, y, w, h)
                
        if box_coords is not None:
            x, y, w, h = box_coords
            y_min, y_max = max(0, y), min(frame_bgr.shape[0], y + h)
            x_min, x_max = max(0, x), min(frame_bgr.shape[1], x + w)
            crop_bgr = frame_bgr[y_min:y_max, x_min:x_max]
            
            if crop_bgr.size > 0:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                pred_label, confidence, probs = predict_cnn(pil_crop, cnn_model, class_names)
                
                # Dibujar bounding box en copia de la imagen original
                marked_img = img_np.copy()
                cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
                st.image(marked_img, caption="Alimento detectado con Bounding Box", use_column_width=True)
            else:
                pred_label, confidence, probs = predict_cnn(image_to_process, cnn_model, class_names)
                st.image(image_to_process, caption="Imagen cargada", use_column_width=True)
        else:
            pred_label, confidence, probs = predict_cnn(image_to_process, cnn_model, class_names)
            st.image(image_to_process, caption="Imagen cargada (No se autodetectó Bounding Box)", use_column_width=True)
            
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

    print("\n==========================================")
    print("Press 'q' to exit OpenCV live stream...")
    print("Consideraciones de Uso de Bounding Box:")
    print("  - Coloca el alimento sobre un plato claro (contraste).")
    print("  - Evita mover excesivamente rápido la cámara.")
    print("==========================================\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Segmentación en tiempo real para extraer Bounding Box
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        _, thresh = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        box_coords = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 2500: # Filtro de ruido
                x, y, w, h = cv2.boundingRect(largest_contour)
                box_coords = (x, y, w, h)

        if box_coords is not None:
            x, y, w, h = box_coords
            y_min, y_max = max(0, y), min(frame.shape[0], y + h)
            x_min, x_max = max(0, x), min(frame.shape[1], x + w)
            crop_bgr = frame[y_min:y_max, x_min:x_max]
            
            if crop_bgr.size > 0:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                label, conf, _ = predict_cnn(pil_crop, cnn_model, class_names)
                
                # Dibujar bounding box verde y overlay
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{label}: {conf * 100:.1f}%"
                cv2.putText(frame, text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Fallback full frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                label, conf, _ = predict_cnn(pil_img, cnn_model, class_names)
                cv2.putText(frame, "No crop area", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Fallback a clasificar frame completo
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            label, conf, _ = predict_cnn(pil_img, cnn_model, class_names)
            
            cv2.putText(frame, "Buscando alimento...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Dibujar un cuadro guía en el centro
            h_f, w_f, _ = frame.shape
            cv2.rectangle(frame, (w_f//4, h_f//4), (3*w_f//4, 3*h_f//4), (0, 0, 255), 1, cv2.LINE_AA)

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
