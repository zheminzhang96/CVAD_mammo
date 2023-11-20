import pydicom 
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

class add_noise():
    def __init__(self) -> None:
        pass


def add_gn_noise(image_path):
    #print("Add gaussian noise to images")
    # read image
    img = cv2.imread(image_path, 0)

    # add gaussian noise 
    gauss_noise=np.zeros(img.shape,dtype=np.uint8)
    cv2.randn(gauss_noise,128,20)
    gauss_noise=(gauss_noise*0.5).astype(np.uint8)
    gn_img=cv2.add(img,gauss_noise)

    return gn_img
    #cv2.imwrite(image_path, gn_img)

        