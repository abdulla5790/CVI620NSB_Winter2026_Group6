import cv2
import numpy as np
from augmentation import augment
from preprocessing import preprocess

# ---------------------------------------------------------------------------
# batch_generator.py
# ---------------------------------------------------------------------------
# This module is responsible for feeding data into the model during training
# and validation. Instead of loading the entire dataset into memory at once
# (which would be impossible for large image datasets), it produces batches
# of images on-the-fly using a Python generator (yield-based function).
#
# Keras' model.fit() accepts generators directly via its steps_per_epoch
# and validation_steps parameters, so this fits cleanly into the training
# pipeline defined in train.py.
# ---------------------------------------------------------------------------


def load_image(data_path, img_path):
    """
    Load a single image from disk given the data root directory and the
    relative or absolute path stored in driving_log.csv.

    The CSV stores full absolute paths from the machine where the data was
    originally collected, so we strip everything except the filename and
    reconstruct the path relative to the current project's data directory.
    This makes the project portable across machines.

    Example:
        img_path  = '/home/user/sim/IMG/center_2023_05_15_12_34_56.jpg'
        filename  = 'center_2023_05_15_12_34_56.jpg'
        full_path = 'data/IMG/center_2023_05_15_12_34_56.jpg'

    OpenCV loads images in BGR channel order by default, so we convert to
    RGB immediately to ensure consistency with the augmentation and
    preprocessing functions (which all expect RGB input).
    """
    # Extract just the filename, discarding any directory prefix from the CSV
    filename = img_path.split('/')[-1]

    # Rebuild the full path relative to the project data directory
    full_path = data_path + '/IMG/' + filename

    img = cv2.imread(full_path)

    # Convert BGR (OpenCV default) -> RGB to match augmentation/preprocessing expectations
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def batch_generator(data_path, image_paths, steerings, batch_size=32, is_training=True):
    """
    Infinite generator that yields (images, steerings) batches for Keras.

    Why a generator instead of loading all data upfront?
        The full image dataset can easily exceed available RAM. A generator
        loads only one batch at a time, keeping memory usage constant
        regardless of dataset size. Keras calls next() on the generator
        automatically at each training step.

    How batches are selected:
        Rather than iterating through the dataset sequentially, we sample
        'batch_size' random indices without replacement (replace=False) on
        every call. This provides random shuffling within each batch and
        avoids the need for a separate shuffle step.

    Training vs. Validation mode (is_training flag):
        - is_training=True  : augmentation IS applied before preprocessing.
          The model sees a different random variation of each image every
          epoch, improving generalisation.
        - is_training=False : augmentation is SKIPPED. Validation images are
          passed through preprocessing only, so the validation loss reflects
          true model performance on unmodified data.

    Preprocessing is always applied (crop -> YUV -> blur -> resize ->
    normalise) regardless of the mode, because the model expects a specific
    input shape and colour space.

    Args:
        data_path    : root directory of the dataset (contains /IMG and CSV)
        image_paths  : array of image path strings from the CSV
        steerings    : array of corresponding steering angle floats
        batch_size   : number of samples per batch (default 32)
        is_training  : whether to apply augmentation (True for train, False for val)

    Yields:
        (np.array of shape [batch_size, 66, 200, 3],
         np.array of shape [batch_size])
    """
    num_samples = len(image_paths)

    # Loop forever — Keras will call next() on this generator indefinitely;
    # steps_per_epoch and validation_steps in model.fit() control when each
    # epoch ends, not this loop.
    while True:
        # Randomly pick 'batch_size' unique indices from the dataset.
        # replace=False ensures no image appears twice in the same batch.
        indices = np.random.choice(num_samples, batch_size, replace=False)

        batch_imgs = []      # Accumulates preprocessed images for this batch
        batch_steerings = [] # Accumulates corresponding steering angles

        for idx in indices:
            # Retrieve the image and its steering label at this index.
            # Example:
            #   image_paths[452] = 'center_2023_05_15_12_34_56.jpg'
            #   steerings[452]   = 0.3245  (32.45% right turn)
            img = load_image(data_path, image_paths[idx])
            steering = steerings[idx]

            # Apply random augmentations only during training.
            # Augmentation may also adjust the steering angle (e.g. flip, rotate).
            if is_training:
                img, steering = augment(img, steering)

            # Preprocess: crop sky/hood -> convert to YUV -> Gaussian blur
            #             -> resize to (200, 66) -> normalise to [0, 1]
            # Output shape: (66, 200, 3) — exactly what the Nvidia CNN expects.
            img = preprocess(img)

            batch_imgs.append(img)
            batch_steerings.append(steering)

        # Convert Python lists to numpy arrays and yield the completed batch.
        # 'yield' pauses here and resumes from this point on the next next() call,
        # preserving the while-True loop state — this is what makes it a generator.
        yield np.array(batch_imgs), np.array(batch_steerings)


# ---------------------------------------------------------------------------
# Usage example (for reference — not executed when imported as a module):
#
#   gen = batch_generator('data', X_train, y_train, batch_size=32, is_training=True)
#
#   # Get first batch (32 randomly selected, augmented, preprocessed images)
#   batch_imgs, batch_steerings = next(gen)
#   # batch_imgs.shape    -> (32, 66, 200, 3)
#   # batch_steerings.shape -> (32,)
#
#   # Get second batch (different random indices each time)
#   batch_imgs, batch_steerings = next(gen)
#
#   # With 10,000 samples and batch_size=32: ~312 batches per epoch.
#   # Keras handles this automatically via steps_per_epoch = len(X_train) // batch_size
# ---------------------------------------------------------------------------
