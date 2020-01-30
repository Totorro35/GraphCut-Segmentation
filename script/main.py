import argparse
import cv2
import numpy as np
import scipy
from scipy.misc import imread
import maxflow
from matplotlib import pyplot as ppl
import math


## Global Var
drawing=False # true if mouse is pressed
mode=True
alpha=0.4
color=(0,0,255)

def wij(x,y,sigma=1) :
    factor= ((x-y)**2)/(2*sigma**2)
    return math.exp(-factor)

def delta_func(x,y) :
    if x==y:
        return 1
    else :
        return 0

def R_func(img,value) :
    return abs(value-img)

def computeSegmentation(path,image):
    F = {}
    B = {}


    img = imread(path,flatten=True)/255

    # Create the graph.
    g = maxflow.Graph[float]()
    # Add the nodes. nodeids has the identifiers of the nodes in the grid.
    nodeids = g.add_grid_nodes(img.shape)

    # Add non-terminal edges with the same capacity.
    #g.add_grid_edges(nodeids, 128)
    for i in range(img.shape[0]-1) :
        for j in range(img.shape[1]-1) :

            #gX = alpha*(1-abs(img[i][j]-img[i+1][j]))**25
            #gY = alpha*(1-abs(img[i][j]-img[i][j+1]))**25

            gX = alpha*delta_func(img[i][j],img[i+1][j])*wij(img[i][j],img[i][j+1])
            gY = alpha*delta_func(img[i][j],img[i][j+1])*wij(img[i][j],img[i][j+1])

            g.add_edge(nodeids[i][j],nodeids[i+1][j],gX,0)
            g.add_edge(nodeids[i][j],nodeids[i][j+1],gY,0)

    g.add_grid_tedges(nodeids, R_func(img,0), R_func(img,0.8))

    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)

    # The labels should be 1 where sgm is False and 0 otherwise.
    result = np.int_(np.logical_not(sgm))

    ppl.imshow(result)
    ppl.show()

def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(image,(current_former_x,current_former_y),(former_x,former_y),color,5)
                current_former_x = former_x
                current_former_y = former_y

    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(image,(current_former_x,current_former_y),(former_x,former_y),color,5)
            current_former_x = former_x
            current_former_y = former_y

    return former_x,former_y

def parseArgument():
    parser = argparse.ArgumentParser()

    ## Database name
    parser.add_argument("-i", "--image", dest="img",
                        help="input image", metavar="STRING", default="None")

    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgument()

    image = cv2.imread(args.img)
    cv2.namedWindow("Graph Cut Segmentation")
    cv2.setMouseCallback('Graph Cut Segmentation',paint_draw)

    while(1):
        cv2.imshow('Graph Cut Segmentation',image)
        k=cv2.waitKey(1)& 0xFF
        if k!=255:
            print(k)

        if k==27: #Escape KEY
            cv2.imwrite("painted_image.jpg",image)
            break

        if k==114: #R Key
            color=(0,0,255)

        if k==98: #B Key
            color=(255,0,0)

        if k==99: #C Key
            computeSegmentation(args.img,image)

    cv2.destroyAllWindows()