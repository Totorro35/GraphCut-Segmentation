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
color_mode=0
F_px={}
B_px={}

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

def balanceWeightsForm(value, meanForm, sigmaForm,meanBack, sigmaBack) :
    pdfForm = pdf(value, meanForm, sigmaForm)
    pdfBack = pdf(value, meanBack, sigmaBack)
    balanceW = pdfForm/(pdfForm+pdfBack)
    return balanceW

def balanceWeightsBack(value, meanForm, sigmaForm,meanBack, sigmaBack) :
    pdfForm = pdf(value, meanForm, sigmaForm)
    pdfBack = pdf(value, meanBack, sigmaBack)
    balanceW = pdfBack/(pdfForm+pdfBack)
    return balanceW

def balanceToweight(weight):
    sum_weight=weight[:,:,0]
    for i in range(1,weight.shape[2]):
        sum_weight += weight[:,:,i]
    sum_weight/=weight.shape[2]
    return -lmbd*np.log(sum_weight)

def wij(x,y,sigma=1) :
    factor= (LA.norm(x-y)**2)/(2*sigma**2)
    return math.exp(-factor)

def computeSegmentation(image):

    img = image

    F_pdf_mean,F_pdf_sigma = computePdf(list(F_px.values()))
    B_pdf_mean,B_pdf_sigma = computePdf(list(B_px.values()))

    # Create the graph.
    g = maxflow.Graph[float]()

    # Add the nodes. nodeids has the identifiers of the nodes in the grid.
    nodeids = g.add_grid_nodes((img.shape[0],img.shape[1]))

    g.add_grid_tedges(nodeids, balanceToweight(balanceWeightsBack(img,F_pdf_mean,F_pdf_sigma,B_pdf_mean,B_pdf_sigma)), balanceToweight(balanceWeightsForm(img,F_pdf_mean,F_pdf_sigma,B_pdf_mean,B_pdf_sigma)))

    for i in F_px :
        y = i[0]
        x = i[1]
        g.add_tedge(nodeids[y][x],infinity,0)

    for i in B_px :
        y = i[0]
        x = i[1]
        g.add_tedge(nodeids[y][x],0,infinity)

    # Add non-terminal edges with the same capacity.
    #g.add_grid_edges(nodeids, 128)
    for i in range(img.shape[0]-1) :
        for j in range(img.shape[1]-1) :

            gX = wij(img[i][j],img[i][j+1])
            gY = wij(img[i][j],img[i][j+1])

            g.add_edge(nodeids[i][j],nodeids[i+1][j],gX,0)
            g.add_edge(nodeids[i][j],nodeids[i][j+1],gY,0)

    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)

    # The labels should be 1 where sgm is False and 0 otherwise.
    result = np.int_(np.logical_not(sgm))

    ppl.imshow(result,cmap="plasma")
    ppl.show()

def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode

    if color_mode==0:
        color = (0,215,255)
    else:
        color = (208,224,64)

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                if color_mode==0:
                    B_px[(former_y,former_x)]=originals_image[former_y][former_x]
                else :
                    F_px[(former_y,former_x)]=originals_image[former_y][former_x]
                cv2.line(image,(current_former_x,current_former_y),(former_x,former_y),color,3)
                current_former_x = former_x
                current_former_y = former_y

    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            if color_mode==0:
                B_px[(former_y,former_x)]=originals_image[former_y][former_x][0]
            else :
                F_px[(former_y,former_x)]=originals_image[former_y][former_x][0]
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

    images = build_img(args.img)
    image = cv2.merge((images[:,:,3],images[:,:,3],images[:,:,3]))
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

        if k==114: #R Key
            color_mode=0

        if k==98: #B Key
            color_mode=1

        if k==99: #C Key
            computeSegmentation(originals_image)

    cv2.destroyAllWindows()