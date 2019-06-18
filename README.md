# AI_colab
## Description des fichiers et répertoires de la repository
Deux ensembles d'images ont été générés, l'un est volumineux (13000 images) et l'autre est relativement léger (1300 images).
Des entrainements sont effectués sur les 2 afin de voir lequel fournit un modèle plus précis.
## Bibliothèques utilisées
Les bibliothèques de tensorflow.keras ont été utilisées et tensorflow.
## Algorithme
L'algorithme utilisé est celui de ce tutoriel https://blog.coast.ai/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a.

[Expliquer plus en détail]

Il s'agit d'un CNN avec 5 layers de convolution et 2 layers dense.

[Différences]
Il y a néanmoins des différences dans le pré-traitement de l'image.

## Comment démarrer la simulation
L'AI rentre en jeux automatiquement après avoir appuyé sur le bouton 'Autonome' de l'interface utilisateur. On suppose que le serveur et le modèle ont déjà été exécutés avant d'avoir démarré le programme Unity.
