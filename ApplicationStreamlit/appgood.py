import streamlit as st
import torch
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, MaxPooling2D, Dense,
    Flatten, Dropout, Activation
)
import time
from PIL import Image, ImageDraw, ImageFont
import os

# Configuration de la page
st.set_page_config(
    layout="wide",
    page_title="Surveillance Intelligente",
    page_icon="👁️"
)

# Patch pour éviter que Streamlit plante en inspectant torch.classes
if not hasattr(torch.classes, "__path__"):
    torch.classes.__path__ = []

# --- Custom CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .stButton>button {
            border: 2px solid #4a90e2;
            border-radius: 20px;
            color: white;
            background: linear-gradient(45deg, #4a90e2, #8b6df2);
            padding: 10px 24px;
            margin: 10px 0;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
            background: linear-gradient(45deg, #3a7bd5, #7b5dd3);
        }
        .stTitle {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .fps-counter {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-family: monospace;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .alert-pulse {
            animation: pulse 1.5s infinite;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()
st.title("👁️ Système de Surveillance Intelligente")
st.markdown("Bienvenue à notre application de surveillance intelligente qui combine détection, prédiction de chute et suivi de somnolence en temps réel.")
st.image(
    r"C:\Users\Alif computer\Desktop\differentTypeOfDataThatIuseForMyTest\FormationOpenCV\projetCamputerVision\projetComputerBonne\surveillance_app\static\uploads\IMAGE1.jpg",
    use_container_width=True
)

st.markdown(
    "<h2 style='text-align: center; color: black;'>Lorsque vous cliquez sur 'Démarrer la surveillance', regardez vers le bas : la surveillance a commencé.</h2>",
    unsafe_allow_html=True
)

# --- Fonction pour texte avec accents ---
def draw_french_text(img, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_size = int(font_scale * 30)
        # Essayez plusieurs polices courantes
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("LiberationSans-Regular.ttf", font_size)
                except:
                    font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Convertir le tuple de couleur BGR à RGB si nécessaire
    if isinstance(color, tuple) and len(color) == 3:
        color = color[::-1]  # Convert BGR to RGB
    
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# --- Custom BatchNormalization ---
class FixedBatchNormalization(BatchNormalization):
    @classmethod
    def from_config(cls, config):
        if isinstance(config.get('axis'), list):
            config['axis'] = config['axis'][0]
        return super().from_config(config)

# --- Modèles ---
custom_objects = {
    "BatchNormalization": FixedBatchNormalization,
    "Conv2D": Conv2D,
    "MaxPooling2D": MaxPooling2D,
    "Dense": Dense,
    "Flatten": Flatten,
    "Dropout": Dropout,
    "Activation": Activation
}

@st.cache_resource
def load_models():
    fall_detection = torch.hub.load(
        'ultralytics/yolov5', 'custom',
        path=r'C:\Users\Alif computer\Desktop\differentTypeOfDataThatIuseForMyTest\FormationOpenCV\projetCamputerVision\projetComputerBonne\yolov5\runs\train\fall_detection2\weights\best.pt',
        force_reload=True
    )
    fall_prediction = tf.keras.models.load_model(
        r"C:\Users\Alif computer\Desktop\differentTypeOfDataThatIuseForMyTest\FormationOpenCV\projetCamputerVision\projetComputerBonne\models\Notebook\best_fall_detection1.keras",
        custom_objects=custom_objects
    )
    drowsiness = tf.keras.models.load_model(
        r"C:\Users\Alif computer\Desktop\differentTypeOfDataThatIuseForMyTest\FormationOpenCV\projetCamputerVision\projetComputerBonne\surveillance_app\models\Drowsiness_model.keras",
        custom_objects=custom_objects
    )
    return fall_detection, fall_prediction, drowsiness

fall_detection_model, fall_prediction_model, drowsiness_model = load_models()

# --- Interface ---
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False

st.sidebar.header("Configuration")
with st.sidebar:
    st.markdown("### 🎚️ Contrôles")
    model_choice = st.radio(
        "Mode de surveillance",
        ["Détection Chute", "Prédiction Chute", "Détection Somnolence", "Surveillance Combinée"],
        index=3
    )
    alert_threshold = st.slider("Seuil d'alerte", 1, 3, 2)

    if st.button("▶️ Démarrer la surveillance", key="start"):
        st.session_state.run_detection = True

    if st.button("⏹️ Arrêter", key="stop"):
        st.session_state.run_detection = False

video_placeholder = st.empty()
status_text = st.empty()

# --- Détection ---
if st.session_state.run_detection:
    cap = cv2.VideoCapture(0)
    last_time = time.time()

    while st.session_state.run_detection:
        ret, frame = cap.read()
        if not ret:
            status_text.warning("Problème de flux vidéo")
            break

        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time

        if model_choice == "Détection Chute":
            results = fall_detection_model(frame)
            annotated_frame = results.render()[0]
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        elif model_choice == "Prédiction Chute":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(frame_rgb, (128, 128)) / 255.0
            preds = fall_prediction_model.predict(
                np.expand_dims(np.repeat(img[np.newaxis], 30, axis=0), axis=0), verbose=0)
            label = "⚠️ Risque de chute!" if preds[0][0] > 0.5 else "✅ Stable"
            annotated_frame = draw_french_text(frame_rgb, label, (50, 50), font_scale=1, 
                                            color=(0, 255, 255) if preds[0][0] <= 0.5 else (0, 0, 255))

        elif model_choice == "Détection Somnolence":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(frame_rgb, (145, 145)) / 255.0
            preds = drowsiness_model.predict(img[np.newaxis], verbose=0)
            label = "😴 Somnolence détectée!" if preds[0][0] > 0.5 else "😊 Éveillé"
            color = (0, 0, 255) if preds[0][0] > 0.5 else (0, 255, 0)
            annotated_frame = draw_french_text(frame_rgb, label, (50, 50), font_scale=1, color=color)

        else:  # Mode combiné
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fall_detection_model(frame)
            fall_detected = any(
                [res['name'] == 'fall' for res in results.pandas().xyxy[0].to_dict(orient='records')])
            fall_confidence = max(
                [res['confidence'] for res in results.pandas().xyxy[0].to_dict(orient='records') if res['name'] == 'fall'],
                default=0)

            img_fall = cv2.resize(frame_rgb, (128, 128)) / 255.0
            preds_fall = fall_prediction_model.predict(
                np.expand_dims(np.repeat(img_fall[np.newaxis], 30, axis=0), axis=0), verbose=0)
            fall_predicted = preds_fall[0][0] > 0.5
            fall_pred_confidence = float(preds_fall[0][0])

            img_drowsy = cv2.resize(frame_rgb, (145, 145)) / 255.0
            preds_drowsy = drowsiness_model.predict(img_drowsy[np.newaxis], verbose=0)
            drowsiness_detected = preds_drowsy[0][0] > 0.5
            drowsiness_confidence = float(preds_drowsy[0][0])

            critical_count = sum([fall_detected, fall_predicted, drowsiness_detected])
            annotated_frame = frame_rgb.copy()
            y_offset = 40

            # Texte avec accents utilisant PIL
            annotated_frame = draw_french_text(
                annotated_frame,
                f"Détection Chute: {'OUI' if fall_detected else 'NON'} ({fall_confidence:.1%})",
                (20, y_offset),
                color=(0, 255, 0) if not fall_detected else (0, 0, 255)
            )
            
            annotated_frame = draw_french_text(
                annotated_frame,
                f"Prédiction Chute: {'PROBABLE' if fall_predicted else 'PEU PROBABLE'} ({fall_pred_confidence:.1%})",
                (20, y_offset + 40),
                color=(0, 255, 0) if not fall_predicted else (255, 255, 0)
            )
            
            annotated_frame = draw_french_text(
                annotated_frame,
                f"Somnolence: {'DÉTECTÉE' if drowsiness_detected else 'NON DÉTECTÉE'} ({drowsiness_confidence:.1%})",
                (20, y_offset + 80),
                color=(0, 255, 0) if not drowsiness_detected else (255, 0, 255)
            )

            # Dessiner une ligne de séparation avec OpenCV
            annotated_frame = cv2.line(annotated_frame, (10, y_offset + 110), (frame.shape[1] - 10, y_offset + 110), (255, 255, 255), 2)

            if critical_count >= alert_threshold:
                annotated_frame = draw_french_text(
                    annotated_frame,
                    f"ALERTE CRITIQUE ({critical_count}/3)",
                    (frame.shape[1] // 2 - 150, y_offset + 150),
                    font_scale=1.5,
                    color=(0, 0, 255)
                )

                if int(time.time() * 2) % 2 == 0:
                    annotated_frame = cv2.rectangle(annotated_frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10),
                                  (0, 0, 255), 5)
            else:
                annotated_frame = draw_french_text(
                    annotated_frame,
                    f"Situation Normale ({critical_count}/3)",
                    (frame.shape[1] // 2 - 100, y_offset + 150),
                    font_scale=1,
                    color=(0, 255, 0)
                )

            # Pour le FPS, on peut garder cv2.putText car il n'y a pas d'accents
            annotated_frame = cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        video_placeholder.image(annotated_frame, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()