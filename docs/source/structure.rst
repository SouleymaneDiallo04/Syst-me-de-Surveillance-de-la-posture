V.Structure du Projet
=====================

.. code-block:: bash

   ├── app.py                      # Interface principale Streamlit
   ├── main.py                     # Point d'entrée principal
   ├── models/                     # Modèles entraînés
   │   ├── yolov5_fall.pt          # Détection de chutes (YOLOv5)
   │   ├── drowsiness_model.h5     # Détection de somnolence
   │   └── fall_prediction.h5      # Prédiction de chute
   ├── notebooks/                  # Notebooks pour entraînement et tests
   │   ├── train_drowsiness.ipynb
   │   ├── train_fall_prediction.ipynb
   │   └── test_yolov5.ipynb
   ├── utils/                      # Utilitaires
   │   └── alert.mp3               # Son d'alerte
   ├── README.md                   # Documentation
   └── requirements.txt            # Dépendances du projet
