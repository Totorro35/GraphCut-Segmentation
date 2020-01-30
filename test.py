import numpy as np
import scipy
from scipy.misc import imread
import maxflow
from matplotlib import pyplot as ppl
import math

def delta_func(x,y) :
    if x==y:
        return 1
    else :
        return 0

def beta_func(x,y,sigma=1) :
    factor= ((x-y)**2)/(2*sigma**2)
    return math.exp(-factor)

def R_func(img,value) :
    return abs(value-img)

#img = imread("data/IRM-mammaire.jpg",flatten=True)/255
#img = imread("data/test.png",flatten=True)/255
img = imread("data/cerveau.jpg",flatten=True)/255

# Create the graph.
g = maxflow.Graph[float]()
# Add the nodes. nodeids has the identifiers of the nodes in the grid.
nodeids = g.add_grid_nodes(img.shape)

alpha=0.4

# Add non-terminal edges with the same capacity.
#g.add_grid_edges(nodeids, 128)
for i in range(img.shape[0]-1) :
    for j in range(img.shape[1]-1) :

        #gX = alpha*(1-abs(img[i][j]-img[i+1][j]))**25
        #gY = alpha*(1-abs(img[i][j]-img[i][j+1]))**25

        gX = alpha*delta_func(img[i][j],img[i+1][j])*beta_func(img[i][j],img[i][j+1])
        gY = alpha*delta_func(img[i][j],img[i][j+1])*beta_func(img[i][j],img[i][j+1])

        g.add_edge(nodeids[i][j],nodeids[i+1][j],gX,0)
        g.add_edge(nodeids[i][j],nodeids[i][j+1],gY,0)

g.add_grid_tedges(nodeids, R_func(img,0), R_func(img,0.8))

# Find the maximum flow.
g.maxflow()
# Get the segments of the nodes in the grid.
sgm = g.get_grid_segments(nodeids)

# The labels should be 1 where sgm is False and 0 otherwise.
img2 = np.int_(np.logical_not(sgm))
# Show the result.

ppl.imshow(img)
ppl.show()
ppl.imshow(img2)
ppl.show()