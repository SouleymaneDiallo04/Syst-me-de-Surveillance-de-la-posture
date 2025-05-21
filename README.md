# Système de Surveillance Intelligent des Comportements à Risque chez les Personnes Âgées

Une application intelligente en temps réel basée sur la **vision par ordinateur** pour détecter les **chutes**, la **somnolence** et **prédire les chutes imminentes**, dans le but d'améliorer la sécurité et la qualité de vie des personnes âgées.

---

##  Description du Projet

Ce projet propose une surveillance **non intrusive** en temps réel, en combinant plusieurs modèles d’intelligence artificielle pour détecter des situations à risque :

-  **Somnolence** (fatigue visuelle via EAR & MAR)
-  **Prédiction de chute** (séquences vidéo)
-  **Détection de chute** (basée sur YOLOv5)

Le système fournit des **alertes instantanées** (visuelles et sonores) pour prévenir les accidents domestiques ou les postures critiques.

---

##  Fonctionnalités

### 🔹 1. Détection de Somnolence
- Analyse des yeux (EAR) et de la bouche (MAR)
- Classification : **Actif** ou **Somnolent**
- Détection en temps réel avec suivi vidéo

### 🔹 2. Prédiction de Chute
- Analyse de séquences vidéo pour prédire les risques
- Utilise des modèles entraînés sur des vidéos 
- Résultat affiché avant qu’une chute ne survienne

### 🔹 3. Détection de Chute
- Utilisation de **YOLOv5** pour détecter les chutes en direct
- Encadrement de la personne avec un label "Fall" ou "Normal"
- Détection rapide dans des vidéos en live ou enregistrées

### 🔹 4. Interface Utilisateur
- Application **Streamlit** intuitive
- Choix entre :
  - Mode **Vidéo** (analyse de fichiers)
  - Mode **Live** (caméra en direct)
- Visualisation des résultats et alertes en temps réel

---

## Organisation du Dépôt

```bash
├── app.py                      # Interface principale Streamlit
├── models/                     # Modèles entraînés
│   ├── yolov5_fall.pt          # Détection de chutes (YOLOv5)
│   ├── drowsiness_model.h5     # Détection de somnolence
│   └── fall_prediction.h5      # Prédiction de chute
├── notebooks/                  # Notebooks pour entraînement et tests
│   ├── train_drowsiness.ipynb
│   ├── train_fall_prediction.ipynb
│   └── test_yolov5.ipynb
├── utils/                     
├── alert.mp3                   # Son d’alerte
├── README.md
└── requirements.txt            # Dépendances du projet
