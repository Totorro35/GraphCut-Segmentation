# IMM - Graph Cut Segmentation

## Sources
[An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision](https://discovery.ucl.ac.uk/id/eprint/13383/1/13383.pdf)  
[PyMaxFlow](https://pmneila.github.io/PyMaxflow/tutorial.html#a-first-example)  
[Source code](https://github.com/pmneila/PyMaxflow/)  
[Cours Graph Cut](http://mickaelpechaud.free.fr/graphcuts.pdf)  
[DataSet](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)

## Install
`pip3 install -r requirements.txt`

## Script
Segmentation par Graph Cut en version multi modale  
Il faut selectionner dans le code si l'on choisit un mode multi-modale, RGB ou Lab.  
`python3 script/script.py -h` pour obtenir les différentes informations.
De même, pour choisir l'image ou les images à segmenter il faut utiliser la commande suivant :
`python3 script/script.py --image pathImage` ou `python3 script/script.py --image pathFolderImages`

Quand l'interface s'ouvre, il faut donner des exemple de zone de chaque classe.
Pour ce faire la touche `r` permet de selectionner un exemple pour la classe 1.
La touche `b` permet elle de sélectionner un exemple pour la classe 2.
Enfin, utilisez la touche `c`pour calculer le résultat.

## Multiclass
Segmentation par GraphCut en version multiclass  
Il s'agit d'une extension du premier script mais ne fonctionne pas completement en version multimodale  

Elle fonctionne comme la partie précédente dans la limite ou les touches utilisées pour choisir les classes sont différentes.
- b : classe 1
- c : classe 2
- d : classe 3
- e : classe 4
- a : calculer le résultat
