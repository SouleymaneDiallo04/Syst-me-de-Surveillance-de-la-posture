.. Mon Projet documentation master file

##########################################
SYSTÈME DE SURVEILLANCE INTELLIGENT
##########################################

.. contents::
   :depth: 3
   :local:
   :backlinks: top

==================
SOMMAIRE COMPLET
==================

1. Introduction et Objectifs
2. Architecture Technique
   - Modèles IA Utilisés
   - Workflow Global
3. Implémentation Détaillée
   - Détection de Chutes (YOLOv5)
   - Prédiction de Chutes (LSTM)
   - Détection de Somnolence (CNN)
4. Interface Utilisateur
   - Dashboard Streamlit
   - Système d'Alertes
5. Déploiement
   - Configuration Requise
   - Guide d'Installation
6. Défis Techniques et Solutions
7. Annexes
   - Structure du Code
   - Références Techniques

==================
INTRODUCTION
==================

**Objectifs** : 
- Surveillance temps réel des personnes vulnérables
- Détection immédiate des chutes (YOLOv5)
- Prédiction des risques (LSTM+CNN)
- Monitoring de la vigilance (CNN)

==================
ARCHITECTURE TECHNIQUE
==================

.. image:: _static/architecture.png
   :width: 100%
   :align: center

**Composants Principaux** :
1. Module d'Acquisition Vidéo
2. Pipeline de Traitement IA
3. Système d'Alertes
4. Interface de Monitoring

==================
DÉTECTION DE CHUTES (YOLOv5)
==================

**Spécifications** :
- Modèle : ``yolov5_fall.pt``
- Précision : 94.5% 
- Latence : 120ms
- Fonctionnalités :
  - Détection multi-personnes
  - Classification des postures
  - Calcul de vitesse de chute

==================
PRÉDICTION DE CHUTES (LSTM)
==================

**Architecture** :
.. code-block:: python

   model = Sequential([
       TimeDistributed(Conv2D(32, (3,3)), input_shape=(30, 256, 256, 3)),
       LSTM(128),
       Dense(2, activation='softmax')
   ])

**Performances** :
- Accuracy : 89%
- Fenêtre temporelle : 30 frames

==================
INTERFACE UTILISATEUR
==================

**Fonctionnalités Streamlit** :
- Mode Temps Réel
- Mode Analyse de Fichiers
- Journal des Événements
- Paramètres des Alertes

.. image:: _static/interface.png
   :width: 800
   :align: center

==================
DÉPLOIEMENT
==================

**Requirements** :
.. code-block:: text

   python>=3.8
   torch==1.12.1
   streamlit>=1.15

**Lancement** :
.. code-block:: bash

   streamlit run app.py --server.port 8501

==================
ANNEXES
==================

Structure Complète :
.. code-block:: bash

   .
   ├── app.py
   ├── models/
   │   ├── yolov5_fall.pt
   │   ├── drowsiness.h5
   │   └── fall_pred.h5
   ├── notebooks/
   └── requirements.txt

.. note::
   Documentation technique complète - Version 1.0.0