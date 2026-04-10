import cv2
import numpy as np
import random

# ---------------------------------------------------------------------------
# augmentation.py
# ---------------------------------------------------------------------------
# This module defines a set of image augmentation techniques that are applied
# randomly during training to artificially increase the diversity of the
# dataset. Augmentation helps the model generalise better to road conditions
# it has never explicitly seen (different lighting, slight camera angles, etc.)
# and reduces the risk of overfitting to the specific routes driven during
# data collection.
#
# Each function either:
#   - transforms only the image  (brightness, zoom, pan)
#   - transforms the image AND adjusts the steering angle (flip, rotate)
#     because those operations physically change the direction the car appears
#     to be heading, so the label must be updated to stay consistent.
# ---------------------------------------------------------------------------


def flip(img, steering):
    """
    Mirror the image horizontally and negate the steering angle.

    A left-turn image becomes a right-turn image after flipping, so the
    steering angle must be multiplied by -1 to keep the label correct.
    This effectively doubles the usable dataset by exploiting the
    left/right symmetry of the road.

    cv2.flip codes:
        0  -> flip vertically  (upside-down)
        1  -> flip horizontally (left-right mirror)  <- used here
       -1  -> flip both axes   (180 degree rotation)

    Example:
        steering = -0.3  (turning left)
        after flip -> image mirrors left/right, new steering = +0.3 (turning right)
    """
    return cv2.flip(img, 1), -steering


def brightness(img):
    """
    Randomly adjust the brightness of the image by scaling the V (Value)
    channel in HSV colour space.

    Why HSV?
        In HSV, brightness is isolated in a single channel (V), so we can
        scale it independently without affecting hue or saturation.
        Doing the same in RGB would require scaling all three channels and
        risks colour shift artefacts.

    The scaling ratio is sampled from the range [0.8, 1.2]:
        ratio = 1.0 + 0.4 * (random() - 0.5)
        random() in [0,1)  ->  (random()-0.5) in [-0.5, 0.5)
        0.4 * that         in [-0.2, 0.2)
        1.0 + that         in [ 0.8, 1.2)

    np.clip ensures pixel values stay within the valid [0, 255] range after
    scaling — without it we could get overflow artefacts.

    Steering angle is NOT changed because brightness does not affect the
    geometry of the road ahead.
    """
    # Convert to float first so that the multiplication does not overflow uint8
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(float)

    ratio = 1.0 + 0.4 * (random.random() - 0.5)  # Random scale in [0.8, 1.2)

    # Apply scaling only to the V channel (index 2) and clip to [0, 255]
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)

    # Convert back to uint8 then back to RGB for the rest of the pipeline
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def zoom(img):
    """
    Zoom into the image by a random scale factor between 1.10 and 1.20
    (i.e. 10-20% zoom), keeping the output the same size as the input.

    How it works:
        cv2.getRotationMatrix2D builds a 2x3 affine matrix that combines
        rotation and scaling around a specified centre point. By setting
        angle=0 we get pure scaling (no rotation). cv2.warpAffine then
        applies the matrix, effectively cropping the outer edges to simulate
        a camera zoom.

    Steering angle is NOT changed because zooming does not alter the
    horizontal direction the car is heading.
    """
    scale = 1 + random.uniform(0.1, 0.2)   # Random zoom factor in [1.10, 1.20]
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # Build a pure-scale affine matrix (angle=0, scale=scale) centred on image
    zoom_matrix = cv2.getRotationMatrix2D(center, 0, scale)

    # Apply the transformation; pixels shifted outside boundary are filled black
    return cv2.warpAffine(img, zoom_matrix, (w, h))


def pan(img):
    """
    Randomly translate (pan) the image in both the horizontal and vertical
    directions by up to +/-10% of the image dimensions.

    The affine translation matrix is:
        [[1, 0, tx],
         [0, 1, ty]]
    where tx and ty are the pixel offsets in x and y respectively.
    The 1s on the diagonal mean no scaling or rotation — only a shift.

    Simulates the car being positioned slightly off-centre in the lane,
    teaching the model to correct its steering from a variety of lateral
    starting positions.

    Steering angle is NOT adjusted here. A more sophisticated implementation
    could add a small steering correction proportional to tx, but for this
    project the augmentation variety alone is sufficient.
    """
    h, w = img.shape[:2]

    # Horizontal shift: up to +/-10% of image width
    tx = random.uniform(-0.1, 0.1) * w
    # Vertical shift: up to +/-10% of image height
    ty = random.uniform(-0.1, 0.1) * h

    # 2x3 translation matrix — no rotation or scaling involved
    transition_matrix = np.float32([[1, 0, tx],
                                    [0, 1, ty]])

    return cv2.warpAffine(img, transition_matrix, (w, h))


def rotate(img, steering):
    """
    Rotate the image by a small random angle (+/-5 degrees) and compensate
    the steering angle accordingly.

    Why adjust the steering angle?
        Rotating the image changes the apparent orientation of the road in
        the frame. If the road appears rotated to the right, the model would
        otherwise think the car needs to steer further right than it actually
        does. We correct by subtracting a fraction of the rotation angle from
        the steering label.

    Steering correction formula:
        new_steering = original_steering + (-angle / 25.0)

        The divisor 25.0 is an empirical scaling factor that converts degrees
        of image rotation into the normalised steering unit [-1, 1].

    Example:
        original_steering =  0.3   (turning right 30%)
        angle             =  5 degrees (image rotated clockwise)
        correction        = -5/25  = -0.2
        new_steering      =  0.3 - 0.2 = 0.1  (less right turn needed)

    The correction ensures the steering label continues to describe what the
    car should do relative to the actual road geometry, not the rotated image.
    """
    angle = random.uniform(-5, 5)   # Random rotation in degrees, range [-5, +5]
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # Affine matrix for rotation only (scale=1 means no zoom applied)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h))
    adjusted_steering = steering + (-angle / 25.0)

    return rotated_img, adjusted_steering


def augment(img, steering):
    """
    Apply a random subset of augmentations to a single training image.

    Each augmentation is applied independently with a 50% probability, so
    any combination — including none or all five — can occur in one call.
    This stochastic approach means the model sees a different variation of
    each image in every epoch, significantly increasing effective dataset
    size without collecting more real driving data.

    Augmentation is intentionally applied ONLY during training (controlled
    by the caller in batch_generator.py). Validation images are kept
    unmodified so that validation loss reflects true real-world performance.

    Args:
        img      : RGB numpy array from the simulator camera
        steering : float, the original steering angle label for this image

    Returns:
        img      : augmented image (numpy array, same shape as input)
        steering : potentially adjusted steering angle (float)
    """
    if random.random() < 0.5:
        img, steering = flip(img, steering)       # Mirror left/right

    if random.random() < 0.5:
        img = brightness(img)                     # Random brightness shift

    if random.random() < 0.5:
        img = zoom(img)                           # Random 10-20% zoom in

    if random.random() < 0.5:
        img = pan(img)                            # Random +/-10% translation

    if random.random() < 0.5:
        img, steering = rotate(img, steering)     # Random +/-5 degree rotation

    return img, steering
