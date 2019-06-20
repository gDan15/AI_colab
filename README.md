**Auteurs : Juan Sebastian Barrera Vega & Gaétan Guru**

# Self riding car #
---
## Description des fichiers et répertoires de la repository ##
Deux ensembles d'images ont été générés, l'un est volumineux (13000 images) et l'autre est relativement léger (1300 images).
Des entrainements sont effectués sur les deux ensembles d'images afin de voir lequel fournit un modèle donnat des prédictions plus précises.
## Bibliothèques utilisées ##
Pour utiliser ce code, il faut installer un certain nombre de librairies.
Si vous utilisez anaconda, vous pouvez utiliser les commandes suivantes:
```python console
conda install -c conda-forge python-socketio
conda install -c conda-forge eventlet 
conda install -c anaconda tensorflow 
conda install -c conda-forge graphviz 
conda install -c conda-forge pydot 
conda install -c conda-forge keras 
```


## Algorithme ##
L'algorithme utilisé est celui de ce tutoriel https://blog.coast.ai/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a.

### Explication ###

Le modèle de base utilise le modèle CNN avec 5 couches de convolution et 2 couches dense.

Nous avons utilisé l'optimiseur 'Adam'.

Voici une représentation de la solution :

![picture alt](https://miro.medium.com/fit/c/1838/551/1*XbuW8WuRrAY5pC4t-9DZAQ.jpeg "Modèle CNN")

Source : https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148

### Différences ###
Ils existent plusieurs différences entre notre solution et la solution du tutoriel.

Nous avons établi 2 solutions :
* L'une consiste à modifier le modèle de la solution de base,
* L'autre consiste à ce concentrer sur les données d'entrées et le prétraitement des images.

Pour la première, le nombre de couches composant le modèle a été réduit à 4 couches.

Pour la deuxième, il s'agissait principalement d'ajuster le nombre d'epochs, steps_per_epoch, la taille du batch et le nombre d'image fournies aux programmes.

```python
  net.fit_generator(_generator(256, X, y), steps_per_epoch=50, epochs=160)
```
Dans les deux cas la fonction *fit_generator* a été utilisée afin d'entraîner le modèle itérativement.

## Comment créer le modèle ##
Le modèle peut être crée grâce au fichier *model_generator.py*. Il faut spécifier le répertoire où se trouve le fichier CSV et le dossier contenant les images pour l'entraînement
```python
  DATACSV= r'C:\Users\...'
```
Il est possible d'utiliser le modèle plus complet avec les 5 couches de convolution et 2 dense en décommentant les lignes respectives sur la classe model

https://github.com/gDan15/AI_colab/blob/d8212ad70c1a1d042890561ca9255199acbfb3d2/model_generator.py#L20

## Comment démarrer la simulation ##
Il faut avant tout exécuter le serveur 'Drive' et s'assurer que le bon modèle enregistrer soit chargé.
```python
  model = model(load=True, shape=None, checkpoint="checkpoints/short_light.h1_4")
```
Il faut introduire le bon répertoire ainsi que le bon nom de modèle pour l'attribut checkpoint de la fonction *model* dans le fichier *drive.py*.
Une fois l'application Unity lancée, l'utilisateur doit appuyer sur le bouton 'Autonome'. L'intelligence artificielle se lancera toute seule une fois que la simulation de l'application est chargée.
