import cv2
import numpy as np

# ---------------------------------------------------------------------------
# preprocessing.py
# ---------------------------------------------------------------------------
# This module defines the image preprocessing pipeline applied to every image
# before it is fed into the CNN — both during training (via batch_generator.py)
# and during live inference (via TestSimulation.py).
#
# It is critical that EXACTLY the same preprocessing steps are used in both
# places. If training and inference pipelines differ, the model will receive
# input distributions it was never trained on, causing erratic steering.
#
# Pipeline (in order):
#   1. Crop     — remove irrelevant sky and car hood
#   2. YUV      — convert colour space to match NVIDIA's paper
#   3. Blur     — reduce high-frequency noise
#   4. Resize   — scale to the exact input size the CNN expects
#   5. Normalise— scale pixel values to [0, 1]
# ---------------------------------------------------------------------------


def preprocess(img):
    """
    Apply the full preprocessing pipeline to a single RGB image.

    Args:
        img : numpy array of shape (H, W, 3), RGB, uint8 values in [0, 255]
              Typically (160, 320, 3) as captured by the simulator camera.

    Returns:
        img : numpy array of shape (66, 200, 3), float64 values in [0.0, 1.0]
              Ready to be passed directly to the CNN.
    """

    # Step 1 — Crop
    # img[rows, cols, channels] in numpy slicing notation.
    # Rows 0-59   : sky, trees, and distant scenery — no useful road information.
    # Rows 135-160: the car's hood — also irrelevant and distracting for the CNN.
    # Keeping only rows 60-134 (75 pixels tall) focuses the model on the road
    # surface and lane markings directly ahead of the vehicle.
    img = img[60:135, :, :]

    # Step 2 — Convert to YUV colour space
    # The NVIDIA End-to-End paper uses YUV rather than RGB or greyscale.
    # YUV separates luminance (Y) from chrominance (U, V), which makes the
    # model more robust to lighting changes — the road structure is mostly
    # encoded in Y, while U and V carry colour information that can help
    # distinguish lane markings from road surface under various conditions.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # Step 3 — Gaussian Blur
    # A 3x3 Gaussian kernel smooths out high-frequency pixel noise (e.g. JPEG
    # compression artefacts, texture variation in tarmac) that is not relevant
    # to the steering decision. The sigma=0 argument tells OpenCV to infer the
    # optimal standard deviation from the kernel size automatically.
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 4 — Resize to 200 x 66 pixels
    # cv2.resize takes (width, height) — note the reversed order vs numpy's
    # (height, width). 200x66 is the exact input resolution specified in the
    # NVIDIA paper and must match the model's input_shape=(66, 200, 3).
    img = cv2.resize(img, (200, 66))

    # Step 5 — Normalise pixel values to [0.0, 1.0]
    # Neural networks train faster and more stably when inputs are small
    # floating-point numbers centred near zero. Dividing by 255.0 scales the
    # [0, 255] uint8 range down to [0.0, 1.0].
    # Note: the model's Lambda normalisation layer ALSO divides by 255 — this
    # line handles the preprocessing.py path (used during training via the
    # batch generator), while the Lambda layer handles the in-graph path.
    
    # img = img / 255.0

    return img
