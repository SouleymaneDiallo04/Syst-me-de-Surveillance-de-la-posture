Architectures et Résultats des Modèles
======================================

Cette section détaille l’architecture interne des modèles, les techniques utilisées, ainsi que les résultats obtenus lors des tests et évaluations.

- Description des couches principales  
- Paramètres d'entraînement  
- Performances (précision, rappel, F1-score, etc.)
----

YOLOv5 et son Architecture
==========================

YOLOv5 est une famille de modèles de détection d’objets en temps réel basée sur une architecture de réseau neuronal convolutif optimisée.  

L’architecture de YOLOv5 comprend principalement :  

- **Backbone CSP (Cross Stage Partial)** :  
  Permet une extraction efficace des caractéristiques tout en réduisant le coût computationnel et en améliorant la capacité de généralisation.  
- **Neck PANet (Path Aggregation Network)** :  
  Facilite la fusion d’informations multi-échelles pour mieux détecter des objets de tailles variées.  
- **Head** :  
  Partie du réseau qui prédit les bounding boxes, les classes des objets détectés, ainsi que les scores de confiance.

Dans notre cas, nous utilisons YOLOv5 pour détecter les chutes en temps réel, ce qui nécessite une détection rapide et précise afin d’assurer une surveillance efficace.

----

### Schéma de l’architecture YOLOv5

![Architecture YOLOv5](C:\Users\Alif computer\computer_vision\docs\build\_static\ImageYolov5Model\architecture.png)

*Insérez ici l’image illustrant l’architecture du modèle YOLOv5. Remplacez `C:\Users\Alif computer\computer_vision\docs\build\_static\ImageYolov5Model\architecture.png` par le chemin relatif ou absolu de votre fichier image.*

----

### Espace pour insérer vos résultats et descriptions personnalisés

*Insérez ici vos propres observations, métriques, graphiques et analyses concernant l’architecture et la performance de YOLOv5 dans votre projet.*
