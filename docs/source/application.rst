Description de l’Application
============================
----

Bienvenue à l’application de Surveillance Intelligente
======================================================

Bienvenue dans cette application  de **Surveillance Intelligente** !  
Cette solution a été conçue pour combiner plusieurs modèles d’intelligence artificielle afin d’assurer la sécurité et la vigilance en temps réel.  
Avec elle, vous pouvez détecter les chutes, prédire les risques de chute, surveiller la somnolence, ou combiner tout cela en un seul écran.

Nous allons vous guider à travers les principales fonctionnalités, l’architecture, et même vous montrer le cœur du code qui fait tourner cette application.

----

Fonctionnalités principales
---------------------------

- **Détection de chutes** avec YOLOv5 : détecte instantanément si une chute est en train de se produire devant la caméra.
- **Prédiction de chutes** avec TensorFlow : analyse les séquences pour anticiper si une chute pourrait arriver.
- **Détection de somnolence** : surveille si une personne montre des signes de fatigue.
- **Mode combiné** : regroupe les trois modèles pour une surveillance complète.
- **Interface Streamlit** : une interface web élégante et interactive.
- **Alertes critiques** : déclenchées si plusieurs modèles détectent un problème.
- **Affichage en direct** : flux vidéo annoté avec textes explicatifs et taux de FPS.
- **Configuration utilisateur** : choix des modes, seuil d’alerte, démarrage/arrêt de la surveillance.

----

Structure générale
------------------

L’application utilise les bibliothèques suivantes :
- Streamlit pour l’interface utilisateur.
- OpenCV pour capturer et traiter les flux vidéo.
- PyTorch (YOLOv5) pour la détection temps réel.
- TensorFlow pour les modèles prédictifs et de somnolence.
- PIL (Pillow) pour dessiner les textes avec accents sur les images.

----

Contenu du code principal
-------------------------

Voici le cœur du code Python avec explications et commentaires intégrés.

.. code-block:: python

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

    # Configuration de la page
    st.set_page_config(
        layout="wide",
        page_title="Surveillance Intelligente",
        page_icon="👁️"
    )

    # Patch pour éviter les erreurs avec torch.classes dans Streamlit
    if not hasattr(torch.classes, "__path__"):
        torch.classes.__path__ = []

    # Injecter du CSS personnalisé pour styliser l’interface
    def inject_custom_css():
        st.markdown(\"\"\"
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
        </style>
        \"\"\", unsafe_allow_html=True)

    inject_custom_css()

    st.title("👁️ Système de Surveillance Intelligente")

    # Fonction pour dessiner du texte français avec accents
    def draw_french_text(img, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2):
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        try:
            font_size = int(font_scale * 30)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        if isinstance(color, tuple) and len(color) == 3:
            color = color[::-1]
        draw.text(position, text, font=font, fill=color)
        return np.array(img_pil)

    # Custom BatchNormalization pour compatibilité Keras
    class FixedBatchNormalization(BatchNormalization):
        @classmethod
        def from_config(cls, config):
            if isinstance(config.get('axis'), list):
                config['axis'] = config['axis'][0]
            return super().from_config(config)

    # Chargement des modèles
    @st.cache_resource
    def load_models():
        fall_detection = torch.hub.load(
            'ultralytics/yolov5', 'custom',
            path='path_to_yolov5_weights.pt',
            force_reload=True
        )
        fall_prediction = tf.keras.models.load_model(
            'path_to_fall_prediction_model.keras',
            custom_objects={"BatchNormalization": FixedBatchNormalization}
        )
        drowsiness = tf.keras.models.load_model(
            'path_to_drowsiness_model.keras',
            custom_objects={"BatchNormalization": FixedBatchNormalization}
        )
        return fall_detection, fall_prediction, drowsiness

    fall_detection_model, fall_prediction_model, drowsiness_model = load_models()

    # Interface utilisateur
    if 'run_detection' not in st.session_state:
        st.session_state.run_detection = False

    st.sidebar.header("Configuration")
    with st.sidebar:
        model_choice = st.radio(
            "Mode de surveillance",
            ["Détection Chute", "Prédiction Chute", "Détection Somnolence", "Surveillance Combinée"],
            index=3
        )
        alert_threshold = st.slider("Seuil d'alerte", 1, 3, 2)
        if st.button(" Démarrer la surveillance"):
            st.session_state.run_detection = True
        if st.button(" Arrêter"):
            st.session_state.run_detection = False

    video_placeholder = st.empty()
    status_text = st.empty()

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

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if model_choice == "Détection Chute":
                results = fall_detection_model(frame)
                annotated_frame = results.render()[0]
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            elif model_choice == "Prédiction Chute":
                img = cv2.resize(frame_rgb, (128, 128)) / 255.0
                preds = fall_prediction_model.predict(
                    np.expand_dims(np.repeat(img[np.newaxis], 30, axis=0), axis=0), verbose=0)
                label = " Risque de chute!" if preds[0][0] > 0.5 else "✅ Stable"
                annotated_frame = draw_french_text(frame_rgb, label, (50, 50), font_scale=1,
                                                  color=(0, 255, 255) if preds[0][0] <= 0.5 else (0, 0, 255))

            elif model_choice == "Détection Somnolence":
                img = cv2.resize(frame_rgb, (145, 145)) / 255.0
                preds = drowsiness_model.predict(img[np.newaxis], verbose=0)
                label = " Somnolence détectée!" if preds[0][0] > 0.5 else " Éveillé"
                color = (0, 0, 255) if preds[0][0] > 0.5 else (0, 255, 0)
                annotated_frame = draw_french_text(frame_rgb, label, (50, 50), font_scale=1, color=color)

            else:
                # Mode combiné
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

                img_drowsy = cv2.resize(frame_rgb, (145, 145)) / 255.0
                preds_drowsy = drowsiness_model.predict(img_drowsy[np.newaxis], verbose=0)
                drowsiness_detected = preds_drowsy[0][0] > 0.5

                critical_count = sum([fall_detected, fall_predicted, drowsiness_detected])
                annotated_frame = frame_rgb.copy()

                annotated_frame = draw_french_text(
                    annotated_frame,
                    f"Détection Chute: {'OUI' if fall_detected else 'NON'}",
                    (20, 40),
                    color=(0, 255, 0) if not fall_detected else (0, 0, 255)
                )
                annotated_frame = draw_french_text(
                    annotated_frame,
                    f"Prédiction Chute: {'PROBABLE' if fall_predicted else 'PEU PROBABLE'}",
                    (20, 80),
                    color=(0, 255, 0) if not fall_predicted else (255, 255, 0)
                )
                annotated_frame = draw_french_text(
                    annotated_frame,
                    f"Somnolence: {'DÉTECTÉE' if drowsiness_detected else 'NON DÉTECTÉE'}",
                    (20, 120),
                    color=(0, 255, 0) if not drowsiness_detected else (255, 0, 255)
                )

            video_placeholder.image(annotated_frame, channels="RGB")

        cap.release()
        cv2.destroyAllWindows()

----

Détail technique du code
========================

Imports et configuration initiale
--------------------------------
.. code-block:: python

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

Ces bibliothèques permettent de :
- Gérer l’interface (Streamlit),
- Manipuler les modèles IA (Torch, TensorFlow),
- Capturer et traiter la vidéo (OpenCV),
- Manipuler les images (PIL),
- Gérer les calculs numériques (Numpy),
- Contrôler le temps (time).

----

Configuration de la page Streamlit
---------------------------------
.. code-block:: python

    st.set_page_config(
        layout="wide",
        page_title="Surveillance Intelligente",
        page_icon="👁️"
    )

Configure l’interface web :
- Affiche la page en mode large,
- Définit le titre et l’icône de l’onglet navigateur.

----

Patch pour compatibilité Torch
------------------------------
.. code-block:: python

    if not hasattr(torch.classes, "__path__"):
        torch.classes.__path__ = []

Corrige un bug potentiel lié à la gestion des classes Torch dans Streamlit.

----

Injection de styles CSS personnalisés
-------------------------------------
.. code-block:: python

    def inject_custom_css():
        st.markdown(\"\"\"
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
        </style>
        \"\"\", unsafe_allow_html=True)

Personnalise l’apparence avec un dégradé de fond et des boutons stylisés.

----

Titre principal et message de bienvenue
---------------------------------------
.. code-block:: python

    inject_custom_css()
    st.title(" Système de Surveillance Intelligente")
    st.markdown("Bienvenue à notre application de surveillance intelligente qui combine détection, prédiction de chute et suivi de somnolence en temps réel.")

Affiche le titre de l’application et un message de bienvenue.

----

Fonction pour écrire du texte en français sur les images
--------------------------------------------------------
.. code-block:: python

    def draw_french_text(img, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2):
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        try:
            font_size = int(font_scale * 30)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        if isinstance(color, tuple) and len(color) == 3:
            color = color[::-1]
        draw.text(position, text, font=font, fill=color)
        return np.array(img_pil)

Cette fonction utilise PIL pour dessiner du texte avec accents, ce qui est utile pour les messages en français.

----

Correction BatchNormalization Keras
----------------------------------
.. code-block:: python

    class FixedBatchNormalization(BatchNormalization):
        @classmethod
        def from_config(cls, config):
            if isinstance(config.get('axis'), list):
                config['axis'] = config['axis'][0]
            return super().from_config(config)

Adapte la classe BatchNormalization pour éviter des erreurs lors du chargement des modèles.

----

Chargement des modèles IA
-------------------------
.. code-block:: python

    @st.cache_resource
    def load_models():
        fall_detection = torch.hub.load(
            'ultralytics/yolov5', 'custom',
            path='path_to_yolov5_weights.pt',
            force_reload=True
        )
        fall_prediction = tf.keras.models.load_model(
            'path_to_fall_prediction_model.keras',
            custom_objects={"BatchNormalization": FixedBatchNormalization}
        )
        drowsiness = tf.keras.models.load_model(
            'path_to_drowsiness_model.keras',
            custom_objects={"BatchNormalization": FixedBatchNormalization}
        )
        return fall_detection, fall_prediction, drowsiness

Charge et met en cache les trois modèles pour la détection chute, la prédiction chute et la somnolence.

----

Interface utilisateur dans la sidebar
-------------------------------------
.. code-block:: python

    st.sidebar.header("Configuration")
    with st.sidebar:
        model_choice = st.radio(
            "Mode de surveillance",
            ["Détection Chute", "Prédiction Chute", "Détection Somnolence", "Surveillance Combinée"],
            index=3
        )
        alert_threshold = st.slider("Seuil d'alerte", 1, 3, 2)
        if st.button(" Démarrer la surveillance"):
            st.session_state.run_detection = True
        if st.button("Arrêter"):
            st.session_state.run_detection = False

Permet à l’utilisateur de configurer :
- Le mode d’analyse,
- Le seuil d’alerte,
- Démarrer ou arrêter la surveillance.

----

Capture vidéo et boucle principale
---------------------------------
.. code-block:: python

    if st.session_state.run_detection:
        cap = cv2.VideoCapture(0)
        last_time = time.time()

        while st.session_state.run_detection:
            ret, frame = cap.read()
            if not ret:
                status_text.warning("Problème de flux vidéo")
                break

Ouvre la caméra et commence à lire les images vidéo en boucle.

----

Calcul des FPS
-------------
.. code-block:: python

    current_time = time.time()
    fps = 1 / (current_time - last_time)
    last_time = current_time

Calcule la vitesse d’affichage en images par seconde.

----

Traitement selon le mode choisi
------------------------------
En fonction du mode, l’image est traitée par le modèle correspondant :

- **Détection Chute :** YOLOv5 analyse l’image pour détecter les chutes.
- **Prédiction Chute :** le modèle TensorFlow prédit le risque de chute.
- **Détection Somnolence :** le modèle TensorFlow détecte la somnolence.
- **Surveillance Combinée :** combine les trois modèles et affiche les résultats en superposition.

Les résultats sont annotés sur l’image affichée en temps réel.

----

Affichage du flux vidéo annoté
------------------------------
.. code-block:: python

    video_placeholder.image(annotated_frame, channels="RGB")

Streamlit affiche la vidéo avec les annotations en direct.

----

Libération des ressources
-------------------------
.. code-block:: python

    cap.release()
    cv2.destroyAllWindows()

Libère la caméra et ferme toutes les fenêtres à la fin de la surveillance.

----
