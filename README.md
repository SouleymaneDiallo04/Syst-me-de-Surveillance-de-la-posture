# camputer_vision
## Système de Détection des Comportements Anormaux chez les Personnes Âgées à l'Aide de la Vision par Ordinateur et de l'IA
Ce projet vise à développer un système intelligent capable de surveiller les personnes âgées, détecter les comportements anormaux tels que les chutes et envoyer des alertes en temps réel pour assurer leur sécurité et leur bien-être.

## Concept
Des caméras capturent des vidéos dans l’espace de vie.

Des algorithmes d’apprentissage automatique analysent ces vidéos pour :

Détecter la présence d’une personne et suivre ses mouvements.

Identifier si la personne est âgée ou non à l’aide d’un modèle de reconnaissance faciale et morphologique.

Reconnaître la posture et l’activité : être debout, assis, allongé, en mouvement ou en train de tomber.

Distinguer une chute d’une action normale : éviter les fausses alertes en différenciant une chute d’un simple mouvement volontaire comme s’asseoir rapidement ou ramasser un objet.

Déterminer la durée d’immobilité après une chute : si la personne ne se relève pas dans un délai prédéfini, une alerte est déclenchée.

## Avantages
✅ Surveillance non intrusive : pas besoin de porter un dispositif, contrairement aux montres connectées.
✅ Améliore l’autonomie des seniors : ils peuvent vivre seuls en toute sécurité.
✅ Réduit les délais d’intervention : en cas de chute, l’alerte permet d’apporter de l’aide immédiatement.
✅ Diminue les coûts de santé : une prise en charge rapide évite des complications graves.

### Fonctionnalités du Système
1. Collecte et Prétraitement des Données
Collecte de vidéos diversifiées incluant différentes postures et chutes simulées par des volontaires.

Reconnaissance des personnes âgées à l’aide de modèles de détection basés sur les traits du visage, la posture, et d’autres caractéristiques physiques.

Segmentation de l’image pour identifier le corps et les membres de la personne suivie.

## 2. Détection des Chutes et Anomalies
Suivi des mouvements du corps en utilisant des techniques de vision par ordinateur (pose estimation avec OpenPose, MediaPipe, ou un modèle CNN).

Classification des postures : Debout, Assis, Allongé, Chute.

Analyse du temps d’immobilité après une chute pour éviter les fausses alertes.

Filtrage avancé des anomalies pour éviter les erreurs d’interprétation (ex : éviter de considérer une personne allongée volontairement comme une chute).

## 3. Système d’Alerte Intelligent
Déclenchement automatique d’une alerte si une chute est détectée et que la personne ne se relève pas après un certain délai.

## Moyens d’alerte :

📞 Appel téléphonique vers un proche ou un service d’urgence.

📩 Envoi d’un e-mail ou SMS avec les images et la localisation du lieu de la chute.

🚨 Alerte sonore pour avertir les personnes présentes dans l’environnement.

## 4. Réduction des Faux Positifs et Amélioration de la Fiabilité
Intégration de modèles robustes (CNN, RNN, Transformer) pour éviter les erreurs.

Entraînement du système avec des données variées pour s’adapter aux différentes morphologies et environnements.

Vérification contextuelle avant de déclencher une alerte (ex : vérifier si la personne tente de se relever).

## Technologies Utilisées
🖥️ Vision par Ordinateur : OpenCV, MediaPipe, OpenPose
📊 Apprentissage Automatique : TensorFlow, PyTorch
📡 Cloud & Alertes : Twilio API (Appels/SMS), SMTP (Email), Firebase (Notifications push)


