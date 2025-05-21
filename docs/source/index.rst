Système de Surveillance Intelligent
===================================

.. contents::
   :depth: 2
   :local:

Introduction
------------

Une application intelligente en temps réel basée sur la **vision par ordinateur** pour détecter :

- Les **chutes**
- La **somnolence** 
- Les **chutes imminentes**

Objectif : Améliorer la sécurité et la qualité de vie des personnes âgées.

Description du Projet
---------------------

Solution de surveillance **non intrusive** combinant :

- Modèles d'IA spécialisés
- Détection en temps réel
- Alertes immédiates

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

Structure du Projet
-------------------

.. code-block:: bash

    ├── app.py                      # Interface Streamlit
    ├── models/
    │   ├── yolov5_fall.pt          # Modèle YOLOv5
    │   ├── drowsiness_model.h5     # Détection fatigue
    │   └── fall_prediction.h5      # Prédiction chute
    ├── notebooks/                  # Entraînement
    ├── utils/                      # Alertes sonores
    └── requirements.txt            # Dépendances

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