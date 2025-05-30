VI.Installation des Packages
============================

Pour assurer le bon fonctionnement des modèles d’intelligence artificielle développés, plusieurs bibliothèques Python doivent être installées. Elles sont regroupées selon les modules fonctionnels : détection de somnolence, prédiction de chute, et détection de chute.

----

Détection de Somnolence
-----------------------

Cette partie utilise principalement **MediaPipe** pour la détection faciale et des landmarks (points de repère), ainsi que des bibliothèques classiques pour le traitement d’image, la manipulation de données, et la construction des modèles de classification.

- **mediapipe** : bibliothèque pour la détection et le suivi des points clés du visage en temps réel.
- **opencv-python** : traitement vidéo et image.
- **numpy, pandas** : manipulation efficace des données.
- **matplotlib** : visualisation des données et des résultats.
- **scikit-learn** : outils pour la création et l’évaluation des modèles ML classiques (SVM, Random Forest, etc.).
- **xgboost** : modèle de gradient boosting pour classification.
- **python-docx, python-pptx** : génération automatique de rapports Word et PowerPoint.

Installation via pip :

.. code-block:: bash

   pip uninstall -y mediapipe
   pip install mediapipe==0.10.7 opencv-python numpy matplotlib pandas scikit-learn xgboost python-docx python-pptx kagglehub

----

Importations utilisées :

.. code-block:: python

   import mediapipe as mp                # Détection des points clés du visage
   import cv2                           # Traitement image/vidéo
   import numpy as np                   # Manipulation des matrices et tableaux
   import matplotlib.pyplot as plt     # Visualisation graphique
   import os
   import pandas as pd                  # Manipulation de données tabulaires
   import shutil
   import pickle                       # Sérialisation des modèles
   import sklearn
   from sklearn.model_selection import train_test_split
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.svm import SVC
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                roc_curve, roc_auc_score, precision_recall_curve,
                                classification_report, confusion_matrix)
   import xgboost as xgb               # Modèle boosting performant
   from docx import Document           # Génération de documents Word
   from pptx import Presentation       # Génération de présentations PowerPoint
   from pptx.util import Inches

----


Prédiction de Chute
-------------------

Cette partie utilise **TensorFlow/Keras** pour entraîner un modèle profond combinant CNN et LSTM afin d’exploiter les séquences vidéo pour anticiper les chutes.

- **tensorflow, keras** : construction, entraînement et évaluation des modèles de deep learning.
- **scikit-learn** : gestion du dataset et métriques.
- **matplotlib** : visualisation des courbes d’entraînement.
- **networkx** : manipulation de graphes, utilisée pour des analyses avancées.

Installation via pip :

.. code-block:: bash

   pip install tensorflow scikit-learn matplotlib numpy keras networkx

----
Importations utilisées :

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

----