import cv2
import numpy as np
import random

# This script defines various image augmentation techniques to enhance the diversity of the training dataset 

# Flip the image horizontally and invert the steering angle
def flip(img, steering):
    #Warp affine is slower than flip, so we use flip for horizontal flipping
    #0 = Flip vertically (upside down)
    #1 = Flip horizontally (left-right mirror)
    #-1 = Flip both axes (180° rotation)

    #If steering angle is -0.3, it becomes 0.3 after flipping
    return cv2.flip(img, 1), -steering

# Adjust the brightness of the image by converting to HSV and scaling the V channel
def brightness(img):
    #Convert the image from RGB to HSV color space, where we can easily manipulate the brightness (V channel)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(float)
    ratio = 1.0 + 0.4 * (random.random() - 0.5) # Randomly adjust brightness by ±20%
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255) # Apply the brightness adjustment to the V channel and clip to valid range
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB) # Convert back to RGB after adjusting brightness

# Zoom the image by scaling it and then cropping back to original size
def zoom(img):
    scale = 1 + random.uniform(0.1, 0.2) # Randomly zoom in by 10-20%
    h, w = img.shape[:2] # Get the height and width of the image
    center = (w / 2, h / 2) # Calculate the center of the image
    #Zoom matrix = cv2.getRotationMatrix2D(center, angle, scale)
    zoom_matrix = cv2.getRotationMatrix2D(center, 0, scale) # Create a scaling transformation matrix centered on the image
    #return zoomed image
    return cv2.warpAffine(img, zoom_matrix, (w, h))

# Pan the image by translating it in both x and y directions
def pan(img):
    h, w = img.shape[:2] # Get the height and width of the image
    #Gets image width and shifts the image horizontally between -10% and +10% 
    tx = random.uniform(-0.1, 0.1) * img.shape[1]
    #Gets image height and shifts the image vertically between -10% and +10% 
    ty = random.uniform(-0.1, 0.1) * img.shape[0]
    #Create a translation matrix for panning the image by tx and ty
    # This creates a 2x3 affine transformation matrix
    # [[1, 0, tx], - no scaling in x direction, shift horizontally by tx pixels
    #  [0, 1, ty]] - no scaling in y direction, shift vertically by ty pixels
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    #return the panned image by applying the translation matrix to the original image 

    return cv2.warpAffine(img, translation_matrix, (w, h))

# Rotate the image by a small angle and adjust the steering angle accordingly
def rotate(img, steering):
    angle = random.uniform(-5, 5) # Randomly rotate the image by ±5 degrees
    h, w = img.shape[:2] # Get the height and width of the image
    center = (w / 2, h / 2) # Calculate the center of the image
    #Rotation matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    #Return the rotated image and adjust the steering angle based on the rotation 
    #Rotating the image means car is turning, so we adjust the steering angle by a factor of the rotation angle
    # Example: Rotate right, original steering right
    # steering = 0.3  # Already turning right 30%
    # angle = 5       # Image rotated right 5°
    # new_steering = 0.3 + (-5/25) = 0.3 - 0.2 = 0.1  # Need less right turn
    #We basically want the steering to match the road and not the car's orientation in the image. 
    return cv2.warpAffine(img, rotation_matrix, (w, h)), steering + (-angle / 25.0)

# Apply a random combination of augmentations to the image and adjust the steering angle as needed
def augment(img, steering):
    #Randomly apply each augmentation with a 50% chance 
    #Could be any combination of the augmentations, including none or all of them
    if random.random() < 0.5:
        img, steering = flip(img, steering)
    if random.random() < 0.5:
        img = brightness(img)
    if random.random() < 0.5:
        img = zoom(img)
    if random.random() < 0.5:
        img = pan(img)
    if random.random() < 0.5:
        img, steering = rotate(img, steering)
    return img, steering