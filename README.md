# Système de Surveillance de la Fatigue et des Comportements Anormaux chez les Personnes Âgées

Une application intelligente en temps réel utilisant la vision par ordinateur et la méthode METEO pour détecter les signes de fatigue et les postures à risque chez les personnes âgées.

## Introduction

Ce projet vise à améliorer la sécurité et la qualité de vie des personnes âgées à travers une surveillance non intrusive en temps réel. Le système détecte :

- La fatigue via les rapports d’aspect des yeux (EAR) et de la bouche (MAR),
- Les comportements ou postures anormaux via une analyse METEO basée sur les angles articulaires (MediaPipe).

Il fournit des alertes instantanées en cas de comportements suspects (chutes, somnolence, postures à risque) afin de prévenir les accidents domestiques ou les troubles musculo-squelettiques.

## Fonctionnalités

### Détection de Fatigue (EAR & MAR)

- Suivi en temps réel des yeux pour détecter la fermeture prolongée (somnolence).
- Analyse de la bouche pour identifier les bâillements.
- Classifie l'état comme : `Actif` / `Fatigué (somnolant)`.

### Analyse Posturale METEO

- Estimation des angles articulaires avec MediaPipe.
- Application de la méthode METEO pour l’évaluation du risque ergonomique.
- Génération d’un score IPO et  IPE pour chaque posture observée.
- Identification des postures à haut risque.

### Alertes Intelligentes

- Alertes visuelles dans l’interface (streamlit).
- Alarmes sonores via Pygame.
- Journalisation des alertes pour analyse historique.

### Interface Utilisateur

- Application Streamlit intuitive.
- Choix entre :
  - Mode Vidéo : analyse de vidéos préenregistrées.
  - Mode Live : analyse directe depuis webcam.

## Fichiers Principaux

```
├── app.py                      # Interface principale Streamlit
├── fatigue_detection.py       # Détection EAR & MAR
├── posture_meteo.py           # Analyse METEO
├── models/
│   ├── svm_fatigue.pkl
│   ├── rf_fatigue.pkl
│   └── mlp_fatigue.pkl
├── alert.mp3                  # Son pour les alertes
├── notebooks/
│   ├── build_models.ipynb
│   └── test_mediapipe.ipynb
├── feats/
│   ├── drowsy.csv
│   └── non_drowsy.csv
├── README.md
└── requirements.txt
```

## Dépendances

- Python 3.8+
- OpenCV (vision par ordinateur)
- MediaPipe (repères articulaires)
- Scikit-learn (modèles ML)
- TensorFlow (modèles CNN si extension future)
- Pygame (alertes sonores)
- Streamlit (interface utilisateur)
- NumPy, Matplotlib, Pandas

## Lancer l’Application

```bash
streamlit run app.py
```

Assurez-vous :

- Que votre webcam est disponible.
- De fermer les logiciels qui l’utilisent avant de démarrer.

## Détails Techniques

### Détection de Fatigue

- EAR = Distance verticale des yeux / Distance horizontale.
- MAR = Distance verticale de la bouche / Distance horizontale.

### Détection par la méthode METEO

- Calcul des angles articulaires à partir des coordonnées MediaPipe.
- Calcul de l’IPE (Indice de Pénibilité Ergonomique) basé sur les scores de flexion.
- Identification des postures dangereuses et durée d’exposition.
- Classification en : Risque Faible, Modéré, Élevé.

## Jeux de Données

### Dataset Fatigue

- Source : Kaggle (Drowsiness Dataset)
- Catégories : Drowsy vs Non-Drowsy
- Utilisé pour entraîner SVM, RF, MLP.

### Dataset Postures (METEO)

- Jeux personnalisés capturés ou extraits de vidéos d’activités simulées.
- Mesure des angles pour : cou, dos, genoux, etc.
- Label manuel des risques selon les recommandations ergonomiques (RULA, REBA).

## Améliorations Futures

- Détection de chutes par séquence vidéo (LSTM ou pose temporal analysis).
- Intégration d'un modèle 3D pour plus de précision (OpenPose, BlazePose 3D).
- Surveillance continue avec historique d’alertes.
- Application mobile pour notifications aux proches ou soignants.
- Personnalisation des seuils par profil d’utilisateur.

## Contributions

Ce projet est en développement. Les contributions sont les bienvenues pour :

- Améliorer les modèles de classification.
- Étendre à d'autres comportements à risque (ex. désorientation, fugue).
- Améliorer la détection de posture dans des environnements complexes.


