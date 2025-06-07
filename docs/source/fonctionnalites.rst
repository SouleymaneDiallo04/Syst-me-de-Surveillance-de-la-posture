IV. Fonctionnalités Principales
===============================
----

.. list-table::
   :widths: 30 70
   :header-rows: 0

----
   * - 🔹 **Détection de Somnolence**
   
----
     - Utilisation d'un modèle CNN pour l'analyse en temps réel du visage. Extraction des indicateurs clés tels que :
       - EAR (Eye Aspect Ratio) pour détecter la fermeture des yeux.
       - MAR (Mouth Aspect Ratio) pour détecter les bâillements.
       - Classification binaire entre "Actif" et "Somnolent".
       - Fonctionne à partir du flux caméra ou de vidéos pré-enregistrées.
       - Adapté à la surveillance des conducteurs ou du personnel médical.

----
   * - 🔹 **Prédiction de Chute**
----

     - Analyse temporelle de vidéos via des séquences d’images de taille fixe (30 frames).
       - Traitement spatial avec CNN pour extraire les caractéristiques visuelles.
       - Traitement temporel avec LSTM pour modéliser les relations entre les images.
       - Modèle CNN+LSTM entraîné sur des vidéos segmentées en phases pré-chute.
       - Classement des séquences en "Activité Normale" ou "Chute Éminente".
       - Pré-traitement automatique : redimensionnement, normalisation, rééchantillonnage.
       - Alerte préventive possible avant que la chute n’ait lieu.

----
   * - 🔹 **Détection de Chute**
----
     - Détection en temps réel à l’aide de **YOLOv5**, un modèle de détection rapide.
       - Identification immédiate de personnes allongées ou dans des postures anormales.
       - Système d’alerte intégré :
         - Affichage de messages d'alerte en overlay.
       - Résultats directement affichés sur le flux vidéo (bounding boxes + labels).
       - Optimisé pour le traitement en local (CPU/GPU).

----
   * - 🔹 **Interface Utilisateur**
----

     - Conçue avec **Streamlit** pour la simplicité, rapidité et interactivité.
       - Interface Web responsive.
       - Deux modes disponibles :
         - **Mode Vidéo** : chargement de fichiers pour traitement batch.
         - **Mode Caméra** : traitement en direct via webcam ou caméra IP.
       - Affichage en temps réel des prédictions sur les vidéos.
       - Option pour démarrer/arrêter l’analyse.
       - Statistiques des modèles (précision, rappel, etc.) affichées en bas de l’interface.
       - Boutons pour exporter les résultats, logs ou alertes.
       - Prise en charge multilingue (prévu).
       - Interface intuitive pour les utilisateurs non techniques.

----
   * - 🔹 **Sécurité et Stockage**
----
     - Authentification de l’utilisateur par identifiants.
       - Accès personnalisé aux vidéos et résultats.
       - Historique des détections consultable.
       - Sauvegarde automatique des journaux d’analyse dans une base de données.
       - Sécurisation des modèles et des routes API prévues lors du déploiement.

----