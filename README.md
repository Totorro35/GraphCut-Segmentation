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

## Multiclass
Segmentation par GraphCut en version multiclass  
Il s'agit d'une extension du premier script mais ne fonctionne pas completement en version multimodale  

