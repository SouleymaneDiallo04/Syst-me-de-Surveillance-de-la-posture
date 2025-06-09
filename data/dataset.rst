Données d'entraînement et de test
=================================

Bonjour et bienvenue cher visiteur !

Vous pouvez trouver les données correspondantes à **l'entraînement et au test** de nos modèles :

- YOLOv5 (modèle de détection de chutes)
- Modèle de prédiction de chutes

Le lien vers les données est disponible ici :

**Lien Mega :** https://mega.nz/folder/5hpAUSDb#XR88CbFDDlc7fNP6Kp0lLw

Détection de somnolence
========================

Les données pour le modèle de **détection de somnolence** peuvent être téléchargées à l'aide du script suivant :

.. code-block:: python

   import kagglehub

   # Télécharger la dernière version du dataset
   path = kagglehub.dataset_download("rakibuleceruet/drowsiness-prediction-dataset")

   print("Path to dataset files:", path)

Ce script utilise le module ``kagglehub`` pour récupérer automatiquement le jeu de données depuis Kaggle.

Utilisation
===========

Ces données sont utilisées pour entraîner nos modèles de vision par ordinateur dans le cadre de notre application de surveillance intelligente.

