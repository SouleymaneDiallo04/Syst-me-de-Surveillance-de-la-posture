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
   configuration  des models
   instalations des packages 

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