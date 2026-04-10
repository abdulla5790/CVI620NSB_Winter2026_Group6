import cv2
import numpy as np
from augmentation import augment
from preprocessing import preprocess

def load_image(data_path, img_path):
    # Handle both absolute and relative paths stored in CSV
    filename = img_path.split('/')[-1] #Example - 'center_2023_05_15_12_34_56.jpg'
    full_path = data_path + '/IMG/' + filename #Example - 'Self Driving Car/data/IMG/center_2023_05_15_12_34_56.jpg'
    img = cv2.imread(full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #openCV loads images in BGR format, convert to RGB for consistency with augmentation and preprocessing
    return img

#
def batch_generator(data_path, image_paths, steerings, batch_size=32, is_training=True):
    #Store total number of images in dataset
    num_samples = len(image_paths)
    while True:
        # Select 32 random indices from 0 to num_samples-1 to create a batch of data
        # replace=False ensures we don't select the same index multiple times in the same batch
        #indices = array of random indices
        indices = np.random.choice(num_samples, batch_size, replace=False)
        
        batch_imgs = [] #List to store the batch of images
        batch_steerings = [] #List to store the corresponding steering angles for the batch of images
        
        #for each index in the selected batch indices
        for idx in indices:
            #load the image in that index and get the corresponding steering angle
            # Example:
            # image_paths[452] = 'center_2023_05_15_12_34_56.jpg'
            # img = load_image('./data', 'center_2023_05_15_12_34_56.jpg')
            # steerings[452] = 0.3245  # 32.45% right turn
            # steering = 0.3245
            img = load_image(data_path, image_paths[idx])
            steering = steerings[idx]
            
            #Apply random augmentations to the image and adjust the steering angle accordingly, but only during training (not during validation/testing)
            if is_training:
                img, steering = augment(img, steering)
            
            # pre-process the image (crop, convertToYUV, blur, resize, normalize) (output shape of image will be (66, 200, 3)
            img = preprocess(img)

            # Add the processed image and adjusted steering angle to the batch lists
            batch_imgs.append(img)
            batch_steerings.append(steering)
        
        # Convert the batch lists to numpy arrays and yield (returns but saves state) them as a batch for training the model
        yield np.array(batch_imgs), np.array(batch_steerings)


'''
Example usage:

Get first batch
batch_1_imgs, batch_1_steerings = next(gen)

# 1. num_samples = 10000
# 2. indices = np.random.choice(10000, 32, replace=False)
#    -> [452, 8912, 123, 6781, 3421, ...] (32 unique indices)
# 3. For idx=452:
#    - Load image_452.jpg
#    - Get steering=0.3245
#    - Augment (maybe flip, rotate, etc.)
#    - Preprocess (crop→YUV→blur→resize→normalize)
#    - Add to batch
# 4. Repeat for all 32 indices
# 5. Yield batch (32 images, 32 steerings)

# Get second batch (different images)
batch_2_imgs, batch_2_steerings = next(gen)
# New random indices selected: [7823, 234, 5671, ...]


(if batchsize =32 and 10000 samples, around 312 batches per epoch)
'''