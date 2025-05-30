Installation des Packages
=========================

Détection de Somnolence
-----------------------

.. code-block:: bash

   pip uninstall -y mediapipe
   pip install mediapipe==0.10.7 opencv-python numpy matplotlib pandas scikit-learn xgboost python-docx python-pptx kagglehub

Importations utilisées :

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

Prédiction de Chute
-------------------

.. code-block:: bash

   pip install tensorflow scikit-learn matplotlib numpy keras networkx

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
