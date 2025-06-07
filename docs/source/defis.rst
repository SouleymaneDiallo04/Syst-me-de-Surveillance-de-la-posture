VIII. Défis Techniques
======================

----

Principaux Challenges
---------------------

----

Ce projet a présenté plusieurs défis techniques majeurs tout au long de son développement, notamment :

- **Optimisation des performances temps réel** : assurer un traitement fluide des vidéos.
- **Réduction des faux positifs** : améliorer la précision du modèle, notamment pour éviter les fausses alertes.
- **Gestion des ressources matérielles** : limitation de la puissance de calcul disponible, notamment lors de l'entraînement ou du déploiement.

----

Difficultés rencontrées
-----------------------

----

1. **Erreur initiale dans le choix du modèle**  
   Au départ, nous avons tenté de construire un modèle de **prédiction de chute** basé sur **YOLOv5**, un modèle conçu pour la détection d’objets. Après deux semaines de tests et d’ajustements, nous avons constaté que ce modèle **n’était pas adapté** à la prédiction temporelle. YOLOv5 est excellent pour détecter des objets dans une image statique, mais **incapable d’apprendre les relations temporelles** nécessaires à la prédiction de chute.

2. **Choix d’une architecture adaptée**  
   Après des recherches approfondies, nous avons opté pour une architecture **CNN + LSTM** :
   - **CNN** pour extraire les caractéristiques spatiales (formes, postures),
   - **LSTM** pour capturer les relations temporelles entre les frames consécutives.

   Cette architecture s’est révélée plus cohérente avec notre besoin, et nous a permis d’obtenir un premier modèle fonctionnel.

3. **Manque de données spécifiques**  
   Le plus grand obstacle a été l’accès à un **jeu de données approprié**. Pour prédire une chute, il faut :
   - Des vidéos montrant **uniquement les activités précédant une chute**.
   - **Exclure les moments de chute ou post-chute**, afin que le modèle apprenne à **anticiper** l’événement.

   Les étapes suivies ont été :
   - Récupération de vidéos humoristiques sur YouTube (chutes simulées par des influenceurs).
   - Découpage avec l’outil **Shotcut** pour ne conserver que les moments avant la chute.
   - Téléchargement de plus de **60 vidéos supplémentaires** depuis le site universitaire :
     `https://fenix.ur.edu.pl/~mkepski/ds/uf.html`.

   Malgré tout, le nombre de vidéos est resté **très faible**, ce qui a limité la **précision du modèle final**.

4. **Perte d’un modèle performant sur GitHub**  
   Après avoir entraîné un premier modèle très performant, nous l’avons hébergé sur **GitHub** sans connaître la politique concernant les fichiers lourds. Résultat : le fichier a été supprimé/modifié automatiquement, ce qui a causé la **perte définitive du modèle**.

    **Leçon apprise** : ne plus stocker de gros fichiers modèles directement sur GitHub sans passer par Git LFS ou un stockage externe.

5. **Incompatibilité de packages avec Flask**  
   Nous avons tenté de construire une interface de démonstration avec **Flask**, mais certains modèles ne se chargeaient pas correctement. Ce problème venait de **conflits entre les versions de packages** (TensorFlow, Keras, OpenCV, etc.) utilisés dans l’environnement Flask.

    Solution :
   - Création d’un **environnement Python séparé** avec les versions exactes compatibles.
   - Conversion des modèles pour les adapter à cet environnement.

6. **Changement de framework pour l’interface utilisateur**  
   Finalement, nous avons abandonné Flask pour adopter **Streamlit**, un framework rapide et interactif pour le prototypage d'applications de data science.

   Résultat : une **interface fonctionnelle et intuitive** a été conçue pour présenter le projet de manière compréhensible aux utilisateurs non techniques.
   
----

Solutions apportées
-------------------

----

- **Quantification et simplification du modèle** : pour accélérer l’inférence.
- **Pipeline de traitement parallélisé** : réduction du temps de prétraitement.
- **Création d’un nouvel environnement Python** : avec des versions de packages compatibles.
- **Adoption de Streamlit** : interface simple à développer, rapide à mettre en œuvre.

----

Challenges ouverts
------------------

----

-  **Développement d’une API REST** :
  - Backend en Python hébergeant les modèles (détection de chute, prédiction, somnolence).
  - Intégration avec une application **Spring Boot** côté client.
  - Sécurisation et **déploiement complet** de l’application.

-  **Création d’un vrai jeu de données pour la prédiction de chute** :
  - Les données disponibles aujourd’hui sont très limitées.
  - Objectif : construire ou annoter un dataset dédié montrant uniquement les **phases pré-chute**.

----

