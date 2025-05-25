.. Mon Projet documentation master file

#######################################
Système de Surveillance Intelligent
#######################################

.. contents::
   :depth: 3
   :local:
   :backlinks: top

==================
Introduction
==================

**Objectif Principal**  
Développer un système de surveillance intelligent pour la sécurité des personnes vulnérables (personnes âgées, patients, etc.) via trois modèles IA complémentaires :

.. _modeles:

Trois Modèles Clés
------------------
1. **Détection de chutes** (YOLOv5) - Identification immédiate
2. **Prédiction de chutes** (LSTM+CNN) - Analyse préventive
3. **Détection de somnolence** (CNN) - Surveillance continue

.. _technique:

Approche Technique
------------------
- **Prétraitement** : Normalisation (redimensionnement, luminosité)
- **Augmentation** : Rotation, flip horizontal
- **Structuration** : Séquences temporelles pour LSTM

----

.. _description:

Description Détaillée
=====================

Fonctionnalités Principales
---------------------------
.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - 🔹 **Détection**
     - YOLOv5 custom (``yolov5_fall.pt``)
   * - 🔹 **Prédiction**
     - LSTM + CNN (``fall_prediction.h5``)
   * - 🔹 **Somnolence**
     - CNN (``drowsiness_model.h5``)
   * - 🔹 **Interface**
     - Streamlit avec modes Vidéo/Temps réel

.. _structure:

Structure du Projet
------------------
.. code-block:: bash

   .
   ├── app.py                  # Interface Streamlit
   ├── models/
   │   ├── yolov5_fall.pt      # Modèle YOLOv5
   │   ├── drowsiness.h5       # Modèle CNN
   │   └── fall_pred.h5        # Modèle LSTM
   ├── notebooks/              # Notebooks d'entraînement
   │   ├── train_drowsiness.ipynb
   │   ├── train_fall_prediction.ipynb
   │   └── test_yolov5.ipynb
   └── requirements.txt        # Dépendances Python

----

.. _implementation:

Implémentation Technique
========================

Workflow Complet
----------------
1. Acquisition vidéo (caméra/RTSP)
2. Détection d'objets (YOLOv5)
3. Analyse comportementale (LSTM)
4. Génération d'alertes (sonores/visuelles)

.. image:: _static/workflow.png
   :width: 800
   :align: center
   :alt: Diagramme du workflow

----

.. _defis:

Défis Techniques
================

Principaux Challenges
---------------------
- Latence temps réel (< 200ms)
- Réduction des faux positifs
- Optimisation GPU/CPU

Solutions Innovantes
-------------------
- Quantification des modèles
- Pipeline parallélisé
- Seuils d'alertes adaptatifs

.. note::
   Documentation mise à jour le |date|. Code source disponible sur `GitHub <https://github.com/votre-repo>`_.