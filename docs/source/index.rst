.. Mon Projet documentation master file

#######################################
Système de Surveillance Intelligent
#######################################
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

Téléchargement et Préparation des Données
-----------------------------------------

Le dataset est téléchargé via kagglehub et copié localement :  

.. code-block:: python

   import kagglehub

   # Télécharger le dataset
   path = kagglehub.dataset_download("rakibuleceruet/drowsiness-prediction-dataset")
   print("Path to dataset files:", path)

   import shutil

   source = "/root/.cache/kagglehub/datasets/rakibuleceruet/drowsiness-prediction-dataset/versions/1"
   destination = "/content/drowsiness_dataset"

   shutil.copytree(source, destination, dirs_exist_ok=True)
   print("Dataset copié dans :", destination)

Importations Utilisées
------------------------

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
------------------

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