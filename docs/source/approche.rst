Approche Technique
==================

Prétraitement des données
-------------------------

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
