Système de Surveillance Intelligent
===================================

.. contents::
   :depth: 2
   :local:

.. _introduction:

============
Introduction
============

**Objectif du Projet**  
Ce système de surveillance intelligente utilise la **vision par ordinateur** pour assurer la sécurité des personnes (personnes âgées, patients, etc.) via trois modèles IA complémentaires :  

----

1. **Détection de chutes** (YOLOv5) : Identifie les chutes en temps réel à partir d'un flux vidéo.  
2. **Prédiction de chutes** (LSTM + CNN) : Anticipe les risques de chute en analysant les séquences temporelles (*LSTM*) et les motifs spatiaux (*CNN*).  
3. **Détection de somnolence** (CNN) : Repère les signes de fatigue (yeux fermés, tête penchée).  

----

**Approche Technique**  
Avant de plonger dans les détails des modèles, nous commençons par le **prétraitement des données**, étape cruciale pour garantir des prédictions fiables. Nos données (vidéos et images annotées) sont :  

----

- **Normalisées** : Redimensionnement, ajustement de luminosité.  
- **Augmentées** : Rotation, flip horizontal pour améliorer la robustesse.  
- **Structurées** : Séparées en séquences temporelles pour le modèle LSTM.  

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
   Pour toute question technique, consulter le dépôt GitHub ou contacter l'équipe projet.