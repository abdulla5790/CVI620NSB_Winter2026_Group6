# Approach & Methodology

**CVI620 — Computer Vision | Final Project**

**Course Instructor:** Ellie Azizi

---

## Overview

This project implements an end-to-end deep learning system for autonomous steering control
in the Udacity self-driving car simulator. A Convolutional Neural Network (CNN) is trained
to map raw front-camera images directly to steering angle predictions — no hand-crafted
lane detection or rule-based logic is involved. The architecture follows NVIDIA's
behavioural cloning approach published in *End to End Learning for Self-Driving Cars*
(Bojarski et al., 2016).

---

## Member 1 — Data Collection, Preprocessing & Augmentation

### Data Collection

Training data was collected by manually driving the simulator in **Training Mode** on
Track 1 (the lake track). The simulator records a front-facing centre camera image and
the corresponding steering angle at approximately 15 frames per second, saving everything
to an `IMG/` folder and a `driving_log.csv` log file.

To ensure broad track coverage and directional balance, the car was driven for
**multiple laps in both forward and reverse directions (7 to be exact in each direction)**. Driving in both directions is
critical — Track 1 is a predominantly left-turning loop, so driving only forward would
produce a dataset heavily biased toward left-steering angles. Reverse laps replicates every
curve, producing the corresponding right-steering examples and preventing the model from
developing a directional preference. The mouse was used for steering rather than the
keyboard to produce smooth, continuous angle values rather than discrete steps.

The final raw dataset contained **21,562 usable samples** (rows with speed > 1 mph,
excluding stationary startup frames) with steering angles ranging from **−0.72 to +0.87**.

### Dataset Balancing

Inspection of the raw steering distribution revealed a severe straight-driving bias:

| Category | Count | Percentage |
|---|---|---|
| Straight (±0.05) | 13,589 | 63.2% |
| Left (< −0.05) | 3,943 | 18.3% |
| Right (> +0.05) | 3,985 | 18.5% |

Nearly 2/3 of all samples had a near-zero steering angle, which indicated the long
straight sections of the track. Training on this distribution would cause the model
to learn that "not steering" is almost always the correct response, leading to poor
turning performance.

To correct this, the steering range was divided into **25 equal-width bins** and any
bin containing more than **400 samples** was randomly subsampled down to that cap. This
reduced the dataset to approximately **3,756 balanced samples** with a near-uniform
distribution across all steering angles, giving the model equal exposure to straight
driving, gentle curves, and sharper turns.

### Preprocessing Pipeline

Every image passes through the following steps before being fed to the model, implemented
in `preprocessing.py`:

1. **Crop (rows 60–135)** — Removes the sky, treeline, and car hood from the frame.
   Only the road surface ahead is retained, eliminating irrelevant visual information
   that could confuse the model.

2. **RGB → YUV colour space** — Converts the image from RGB to YUV, matching the
   colour space used in the original NVIDIA paper. YUV separates luminance (Y) from
   chrominance (U, V), which makes road features more salient and consistent across
   varying lighting conditions compared to RGB.

3. **Gaussian blur (3×3 kernel)** — Applies a mild blur to reduce high-frequency noise
   and small pixel-level variations that are irrelevant to steering decisions.

4. **Resize to 200×66 pixels** — Scales the cropped image to the exact input dimensions
   required by the NVIDIA architecture (width=200, height=66).

5. **Normalisation (÷ 255) (not used)** — Pixel values are normalised from [0, 255] to [0.0, 1.0]
   inside the model's Lambda layer rather than in the preprocessing function. This ensures
   the normalisation is baked into the saved model and applied automatically during
   inference in `TestSimulation.py` without any additional step.

### Data Augmentation

To improve generalisation expand the effective dataset size, five
augmentation techniques were applied **randomly** to each image during training only
(never on the validation set), implemented in `augmentation.py`. Each augmentation is
applied independently with a 50% probability per sample:

| Augmentation | Description | Steering Adjustment |
|---|---|---|
| **Flip** | Horizontally mirrors the image | Angle multiplied by −1 |
| **Brightness** | Scales V channel in HSV by ±20% | None |
| **Zoom** | Scales image in by 10–20% from centre | None |
| **Pan** | Translates image ±10% horizontally and vertically | None |
| **Rotate** | Rotates image ±5° | Adjusted by −angle/25 |

Flipping is especially important: it doubles the effective dataset size and directly
counteracts the left-turn bias by generating mirrored right-turn examples.
The steering angle sign is inverted on flip to remain physically consistent with the
mirrored road geometry. Similarly, rotation adjustments correct the steering label to
match the rotated road orientation rather than the original car heading.

Augmentations are applied randomly rather than uniformly so the model never memorises
a fixed augmented version of any image, maximising the diversity of what it sees across
epochs.

---

## Member 2 — Model Architecture, Training & Testing

### Model Architecture

The NVIDIA End-to-End CNN was selected because it was designed specifically for this
exact task: mapping raw camera pixels to steering angles without any intermediate
representation. It has been validated on real vehicles and is well-suited to the
complexity of the Udacity simulator's single-track environment.

![alt text](model_img.png)

The network is defined in `model.py` and consists of:

- **Lambda normalisation layer** — scales input pixels from [0, 255] to [0.0, 1.0]
  inside the compute graph, so normalisation is applied automatically at inference time.
- **Five convolutional layers** — the first three use 5×5 kernels with stride 2 to
  simultaneously extract features and downsample the spatial dimensions. The final two
  use 3×3 kernels with stride 1 to capture finer spatial detail in the reduced feature
  maps. All convolutional layers use **ELU activation**.
- **Flatten layer** — converts the 3D feature maps to a 1D vector of 1,164 values.
- **Dropout (0.5)** — randomly disables 50% of neurons during training to prevent
  overfitting. Dropout is automatically disabled during inference.
- **Three fully connected layers** (100 → 50 → 10 neurons, ELU activation) —
  progressively compress the feature vector into a steering prediction.
- **Output layer** (1 neuron, no activation) — produces the continuous steering angle
  as a linear regression output.

**ELU was chosen over ReLU** because ReLU outputs exactly zero for all negative inputs,
which can cause neurons to permanently stop contributing gradients (the "dying ReLU"
problem). ELU outputs a small negative value for negative inputs, keeping all neurons
active and pushing mean activations closer to zero — leading to faster and more stable
convergence on this regression task.

**No activation** was used on the output layer because the steering angle is a
continuous real-valued target in [−1, 1]. Adding sigmoid or tanh would unnecessarily
restrict the output range; a linear output is the correct choice for regression.

### Training

The model was trained using `train.py` with the following configuration:

| Hyperparameter | Value | Reasoning |
|---|---|---|
| Optimizer | Adam (lr = 0.001) | Adapts learning rate per parameter; faster convergence than SGD |
| Loss function | Mean Squared Error | Standard regression loss; penalises large errors more than small ones |
| Batch size | 32 | Balances gradient stability and training speed |
| Epochs | 30 | Sufficient for convergence without overfitting given the dataset size |
| Train/Val split | 80% / 20% | Standard split; validation set never augmented |

The dataset was split 80/20 into training and validation sets using a fixed random seed
(`random_state=42`) for reproducibility. The training generator applied random
augmentation on every batch; the validation generator served unmodified images to give
an accurate measure of real-world performance.

The final loss curve showed both training and validation loss converging smoothly to
approximately **0.01 MSE** by epoch 30, with validation loss consistently below training
loss — indicating no overfitting and good generalisation. The trained model was saved as
`model.h5`.

### Autonomous Testing

Testing was performed using `TestSimulation.py`, a Flask-SocketIO server that connects
to the simulator's Autonomous Mode over a local WebSocket on port 4567. For each frame
received from the simulator:

1. The base64-encoded image is decoded and converted to a NumPy array
2. The same preprocessing pipeline (crop, YUV, blur, resize) is applied
3. The model predicts a steering angle
4. Throttle is computed dynamically as `1.0 − speed / maxSpeed`, acting as a cruise
   controller that accelerates when below the target speed and brakes when above it
5. Both values are emitted back to the simulator via the `steer` event

The maximum speed was set to **10 mph** to give the model sufficient reaction time on
curves, particularly at the bridge entrance where lighting and road texture change
abruptly.

---

## Challenges

- **Member 1 — Double normalisation bug:** Early test runs produced a constant steering
  output of approximately 1.19 regardless of the input image. This was traced to the
  model's Lambda layer (÷255) being applied on top of an already-normalised image
  (also ÷255 in `TestSimulation.py`), causing the model to receive near-zero pixel
  values completely outside its training distribution. Removing the redundant division
  from `TestSimulation.py` resolved the issue immediately.

- **Member 2 - Program Incompatibility:** One challenge on my side was that the program did not run properly on my device, so I was not able to test the simulator locally. Because of that, Member 1 handled the actual testing and execution environment. My main contribution was developing the model.py and train.py files. Since the project structure and function definitions were already clearly laid out, I was able to implement the CNN architecture and training pipeline by following the expected format without needing to run the simulator myself. Because the training images and supporting functions were already available, my task was mainly to complete and push the model and training code. Member 1 then ran the training module on their side, and it worked as expected.
  

---

# Video Submission:
[text](https://youtu.be/_lU_HQS-bes)

## References

- Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., ...
  & Zieba, K. (2016). *End to End Learning for Self-Driving Cars.* NVIDIA Corporation.
  [arXiv:1604.07316](https://arxiv.org/abs/1604.07316)
- Udacity Self-Driving Car Simulator:
  https://github.com/udacity/self-driving-car-sim