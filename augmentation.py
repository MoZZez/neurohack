import numpy as np
import cv2

def rotate(img):
    angle=90
    center=(img.shape[0]/2,img.shape[1]/2)
    map_matr=np.empty((2,3),dtype=float)
    rm=cv2.getRotationMatrix2D(center=center,angle=angle,scale=1)

    dst = cv2.warpAffine(img,rm,(img.shape[0],img.shape[1]))
    return dst
