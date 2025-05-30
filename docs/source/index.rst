Bienvenue à la documentation du projet
========================================

----


.. image:: ../_static/ENSAMMeknès.png
   :alt: Système de Surveillance Intelligent
   :align: center
   :width: 100%

Système de Surveillance Intelligent
----------------------------------------


----



.. contents::
   :depth: 2
   :local:

.. _introduction:

============
Introduction
============

**Objectif du Projet**

Ce projet consiste à développer un **système de surveillance intelligent** basé sur la **vision par ordinateur**, visant à assurer la sécurité des personnes vulnérables (telles que les personnes âgées ou les patients) à l’aide de trois modèles d’intelligence artificielle complémentaires.

----

1. **Détection de chutes** (YOLOv5) : Identifie les chutes en temps réel à partir d'un flux vidéo.  
2. **Prédiction de chutes** (LSTM + CNN) : Anticipe les risques de chute en analysant les séquences temporelles (*LSTM*) et les motifs spatiaux (*CNN*).  
3. **Détection de somnolence** (CNN) : Repère les signes de fatigue (yeux fermés, tête penchée).  

----

**Approche Technique**  
Avant de plonger dans les détails des modèles, nous commençons par le **prétraitement des données**, étape cruciale pour garantir des prédictions fiables. Nos données (vidéos et images annotées) sont :  

---

==================
Approche Technique
==================

**Prétraitement des données** :

- **Normalisation**  
  ∙ Redimensionnement des images  
  ∙ Ajustement de la luminosité et contraste  
  ∙ Normalisation des valeurs pixel [0-1]

- **Augmentation**  
  ∙ Rotation aléatoire (±15°)  
  ∙ Flip horizontal  
  ∙ Variation de saturation  

- **Structuration**  
  ∙ Découpage en séquences de 30 frames  
  ∙ Pas temporel de 5 images  
  ∙ Format (séquences, hauteur, largeur, canaux)

----  

Objectif : Améliorer la sécurité et la qualité de vie des personnes âgées.

----

Description du Projet
---------------------

Solution de surveillance **non intrusive** combinant :

- Modèles d'IA spécialisés  
- Détection en temps réel  
- Alertes immédiates  

----

Fonctionnalités Principales
---------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - 🔹 **Détection de Somnolence**
     - Analyse des yeux (EAR) et bouche (MAR) - Classification Actif/Somnolent
   * - 🔹 **Prédiction de Chute**
     - Analyse vidéo préventive - Modèles séquentiels
   * - 🔹 **Détection de Chute**
     - YOLOv5 pour identification immédiate - Alertes visuelles/sonores
   * - 🔹 **Interface Utilisateur**
     - Application Streamlit avec modes Vidéo/Live

----

Structure du Projet
-------------------

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

----

Configuration des Models
------------------------

----

Instalations des Packages
-------------------------

Installation des Packages Essentiels
====================================

**modèle de détection de somnolence**
-------------------------------------

Les installations suivantes sont cruciales pour assurer le bon fonctionnement du **modèle de détection de somnolence**. Elles couvrent la récupération des données, le traitement d’image, l’analyse statistique et les modèles machine learning.

Installation Globale
---------------------

Pour installer tous les packages nécessaires, vous pouvez exécuter :  

.. code-block:: bash

   pip uninstall -y mediapipe
   pip install mediapipe==0.10.7 opencv-python numpy matplotlib pandas scikit-learn xgboost python-docx python-pptx kagglehub

Importations Utilisées
----------------------

Une fois les packages installés, nous les utilisons via les importations suivantes :  

.. code-block:: python

   import mediapipe as mp
   import cv2
   import numpy as np
   import matplotlib.pyplot as plt
   import os
   import pandas as pd
   import shutil
   import pickle
   import sklearn
   from sklearn.model_selection import train_test_split
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.svm import SVC
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve, classification_report, confusion_matrix
   import xgboost as xgb
   from docx import Document
   from pptx import Presentation
   from pptx.util import Inches

Rôle des Packages
-----------------

- **mediapipe** : Détection et suivi des repères faciaux.
- **opencv (cv2)** : Traitement vidéo et image.
- **numpy** : Calcul numérique.
- **matplotlib** : Visualisation de données et courbes.
- **pandas** : Manipulation et analyse de données tabulaires.
- **shutil** : Manipulation des fichiers et répertoires.
- **pickle** : Sérialisation et sauvegarde d’objets Python.
- **scikit-learn (sklearn)** : Prétraitement, modèles SVM/RandomForest, métriques.
- **xgboost** : Modèles d’ensemble performants pour la classification.
- **python-docx** : Lecture et écriture de documents Word.
- **python-pptx** : Création automatique de présentations PowerPoint.
- **kagglehub** : Téléchargement des datasets depuis Kaggle.

----

----

**Modèle de Prédiction de Chute**
---------------------------------

Installation des Packages
-------------------------

Pour assurer le bon fonctionnement du modèle de prédiction de chute, plusieurs packages Python sont indispensables.  
Il est recommandé d’installer ces packages via la commande suivante :

.. code-block:: bash

   pip install tensorflow scikit-learn matplotlib numpy keras networkx

Import des Packages
-------------------

Les imports suivants sont utilisés dans le code du modèle de prédiction de chute :

.. code-block:: python

   import cv2
   import os
   import numpy as np
   from sklearn.model_selection import train_test_split
   from keras.utils import to_categorical
   from keras.models import Sequential
   from keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
   from keras.layers import Dropout, BatchNormalization, GlobalMaxPooling2D
   from keras.callbacks import EarlyStopping
   import matplotlib.pyplot as plt
   import networkx as nx
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, LSTM, Dense
   from tensorflow.keras.layers import Dropout, BatchNormalization, GlobalMaxPooling2D
   from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
   from tensorflow.keras.regularizers import l2
   from tensorflow.keras.optimizers import Adam
   from sklearn.utils import class_weight

Description des Packages
------------------------

- **cv2 (OpenCV)** : Traitement d’images et vidéos, extraction et manipulation de frames.  
- **os** : Gestion des chemins de fichiers et interactions système.  
- **numpy** : Calculs numériques et manipulation de matrices.  
- **sklearn.model_selection.train_test_split** : Division des données en ensembles d’entraînement et de test.  
- **keras.utils.to_categorical** : Conversion des labels en format one-hot encoding.  
- **keras.models.Sequential** : Construction séquentielle des couches du réseau de neurones.  
- **keras.layers (TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, BatchNormalization, GlobalMaxPooling2D)** :  
  - TimeDistributed : Applique une couche à chaque frame d’une séquence vidéo.  
  - Conv2D, MaxPooling2D : Extraction de caractéristiques spatiales dans les images.  
  - Flatten : Aplatissement des tenseurs pour passer d’une couche convolutive à une couche dense.  
  - LSTM : Capture la dynamique temporelle dans les séquences.  
  - Dense : Couche pleinement connectée.  
  - Dropout, BatchNormalization : Techniques de régularisation pour améliorer la généralisation.  
  - GlobalMaxPooling2D : Réduction dimensionnelle en conservant les informations importantes.  
- **keras.callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)** :  
  - EarlyStopping : Arrêt automatique de l’entraînement si la performance ne s’améliore plus.  
  - ModelCheckpoint : Sauvegarde automatique du meilleur modèle.  
  - ReduceLROnPlateau : Réduction du taux d’apprentissage en cas de stagnation.  
- **tensorflow.keras.regularizers.l2** : Régularisation L2 pour éviter le surapprentissage.  
- **tensorflow.keras.optimizers.Adam** : Optimiseur adaptatif efficace.  
- **sklearn.utils.class_weight** : Gestion du déséquilibre des classes en pondérant les exemples.  
- **matplotlib.pyplot** : Visualisation graphique des courbes d’apprentissage et résultats.  
- **networkx** : Analyse et visualisation de graphes (utilisé selon contexte).

Installation Globale Recommandée
--------------------------------

Pour installer rapidement l’ensemble des packages nécessaires au modèle de prédiction de chute, utiliser :

.. code-block:: bash

   pip install tensorflow keras scikit-learn matplotlib numpy opencv-python networkx

----

Documentation Technique
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: Contenu Détaillé:
   
   models
   architectures
   application
   defis
   data 

.. note::
   Pour toute question technique, consulter le dépôt GitHub ou contacter l'équipe projet au mail suivant hinimdoumorsia@gmail.com/.

==================
Structure du Projet
==================

.. code-block:: bash

   ├── app.py                      # Interface principale Streamlit  
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
   └── requirements.txt            # Dépendances du projet

==================
Défis Techniques
==================

**Principaux Challenges** : 

- Optimisation des performances temps réel  
- Réduction des faux positifs  
- Gestion des ressources matérielles  

**Solutions** : 

- Quantification des modèles  
- Pipeline parallélisé  
- Sélection optimale des seuils  

.. note::
   Documentation mise à jour le |date|. Code source disponible sur `GitHub <https://github.com/votre-repo>`_.
