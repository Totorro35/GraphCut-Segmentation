import argparse
import cv2
import numpy as np
import scipy
from scipy.misc import imread
import maxflow
from matplotlib import pyplot as ppl
import math
from numpy import linalg as LA
import glob

## Global Var
drawing=False # true if mouse is pressed
mode=True
lmbd=27
infinity=1000000

## Recup Pixel
current_class=0
Nb_class=2
Px=[]
for i in range(Nb_class):
    Px.append({})

def BGRtoLalphabeta(img_in):
    split_src = cv2.split(img_in)
    L = 0.3811*split_src[2]+0.5783*split_src[1]+0.0402*split_src[0]
    M = 0.1967*split_src[2]+0.7244*split_src[1]+0.0782*split_src[0]
    S = 0.0241*split_src[2]+0.1288*split_src[1]+0.8444*split_src[0]

    L = np.where(L == 0.0, 1.0, L)
    M = np.where(M == 0.0, 1.0, M)
    S = np.where(S == 0.0, 1.0, S)

    _L = (1.0 / math.sqrt(3.0)) * ((1.0000 * np.log10(L)) + (1.0000 * np.log10(M)) + (1.0000 * np.log10(S)))
    Alph = (1.0 / math.sqrt(6.0)) * ((1.0000 * np.log10(L)) + (1.0000 * np.log10(M)) + (-2.0000 * np.log10(S)))
    Beta = (1.0 / math.sqrt(2.0)) * ((1.0000 * np.log10(L)) + (-1.0000 * np.log10(M)) + (-0.0000 * np.log10(S)))

    img_out = cv2.merge((_L, Alph, Beta))
    return img_out

def computePdf(listpxl) :
    mean = np.mean(listpxl)
    sigma = np.std(listpxl)
    return mean, sigma

def pdf(value, mean, sigma) :
    sigma+=1/infinity
    expo = np.exp(-0.5*((value-mean)/sigma)**2)
    pdf = (1/(sigma*np.sqrt(2*3.14)))*expo
    return pdf

def computeWeight(img,means,sigmas):
    pdf_value=[]
    for k in range(Nb_class):
        pdf_value.append(pdf(img,means[k],sigmas[k]))

    sum_pdf = np.sum(pdf_value,axis=0)
    weight = pdf_value/sum_pdf
    sum_weight=weight[:,:,:,0]
    for i in range(1,weight.shape[3]):
        sum_weight += weight[:,:,:,i]

    return -lmbd*np.log(sum_weight)

def wij(x,y,sigma=1) :
    factor= (LA.norm(x-y)**2)/(2*sigma**2)
    return math.exp(-factor)

def computeSegmentation(image):
    img = image

    means=[]
    sigmas=[]
    for i in range(Nb_class):
        mean,sigma = computePdf(list(Px[i].values()))
        means.append(mean)
        sigmas.append(sigma)

    # Create the graph.
    g = maxflow.Graph[float]()

    nodeids = g.add_grid_nodes((Nb_class-1,img.shape[0],img.shape[1]))
    for k in range(Nb_class-1):
        for i in range(img.shape[0]-1) :
            for j in range(img.shape[1]-1) :

                gX = wij(img[i][j],img[i][j+1])
                gY = wij(img[i][j],img[i][j+1])

                g.add_edge(nodeids[k][i][j],nodeids[k][i+1][j],gX,0)
                g.add_edge(nodeids[k][i][j],nodeids[k][i][j+1],gY,0)

    for k in range(Nb_class-1):
        weight = computeWeight(img,means,sigmas)
        if k == 0:
            g.add_grid_tedges(nodeids[k],weight[k],infinity)
        elif k == Nb_class-1:
            g.add_grid_tedges(nodeids[k-1],infinity,weight[k])
        else:
            for i in range(img.shape[0]) :
                for j in range(img.shape[1]) :
                    g.add_edge(nodeids[k-1][i][j],nodeids[k][i][j],weight[k][i][j],infinity)

    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)

    print(sgm.shape)

    # The labels should be 1 where sgm is False and 0 otherwise.
    result = np.int_(np.logical_not(sgm))
    result = np.sum(result,axis=0)

    ppl.imshow(result,cmap="plasma")
    ppl.show()

def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode

    if current_class==0:
        color = (165,116,34)
    elif current_class==1:
        color = (3,92,247)
    elif current_class==2:
        color = (15,196,241)
    elif current_class==3:
        color = (104,3,217)
    elif current_class==4:
        color = (102,204,0)

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                Px[current_class][(former_y,former_x)]=originals_image[former_y][former_x][0]
                cv2.line(image,(current_former_x,current_former_y),(former_x,former_y),color,3)
                current_former_x = former_x
                current_former_y = former_y

    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            Px[current_class][(former_y,former_x)]=originals_image[former_y][former_x][0]
            cv2.line(image,(current_former_x,current_former_y),(former_x,former_y),color,3)
            current_former_x = former_x
            current_former_y = former_y

    return former_x,former_y

def parseArgument():
    parser = argparse.ArgumentParser()

    ## Database name
    parser.add_argument("-i", "--image", dest="img",
                        help="input image", metavar="STRING", default="None")

    return parser.parse_args()

def build_img(dir_path):
    images = []
    for path in glob.glob(dir_path+"*.png"):
        n = cv2.imread(path,0)
        images.append(n)

    for path in glob.glob(dir_path+"*.jpg"):
        n = cv2.imread(path,0)
        images.append(n)

    return np.dstack(tuple(images))

if __name__ == "__main__":
    args = parseArgument()

    if(False):
        images = build_img(args.img)
        image = cv2.merge((images[:,:,3],images[:,:,3],images[:,:,3]))
    elif(False) :
        image = cv2.imread(args.img)
        images = image/255.
    else :
        image = cv2.imread(args.img)
        images = BGRtoLalphabeta(image)

    cv2.namedWindow('Graph Cut Segmentation')
    cv2.resizeWindow('Graph Cut Segmentation', (1000,1000))
    cv2.setMouseCallback('Graph Cut Segmentation',paint_draw)
    global originals_image
    #originals_image = BGRtoLalphabeta(image)
    originals_image = images/255.

    while(1):
        cv2.imshow('Graph Cut Segmentation',image)
        k=cv2.waitKey(1)& 0xFF

        if k==27: #Escape KEY
            cv2.imwrite('painted_image.jpg',image)
            break

        if k==98: #B Key
            current_class=0

        if k==99: #C Key
            current_class=1

        if k==100 and Nb_class>2: #D Key
            current_class=2

        if k==101 and Nb_class>3: #E Key
            current_class=3

        if k==102 and Nb_class>4: #F Key
            current_class=4

        if k==97: #A Key
            computeSegmentation(originals_image)

    cv2.destroyAllWindows()