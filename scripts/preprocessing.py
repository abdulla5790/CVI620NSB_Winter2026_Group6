import cv2
import numpy as np
import matplotlib.pyplot as plt
def preprocess(img):
    # img[rows, cols, channels]
    img = img[60:135, :, :] # crop to road
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) #convert to YUV color space
    img = cv2.GaussianBlur(img, (3, 3), 0) # apply gaussian blur (3x3 kernel, sigma=0)
    #(200 width, 66 height)
    img = cv2.resize(img, (200, 66)) # Resize the image to 200x66 pixels, as used in NVIDIA's model 
    # img = img / 255.0 # normalize - model handles this
    return img
