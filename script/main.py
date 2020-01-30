import argparse
import cv2
import numpy as np

drawing=False # true if mouse is pressed
mode=True

color=(0,0,255)

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
            #Compute

    cv2.destroyAllWindows()