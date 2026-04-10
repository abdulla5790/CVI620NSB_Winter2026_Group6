from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, Flatten, Dense,
                                      Dropout, Lambda)
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------
# Implements the NVIDIA End-to-End Self-Driving CNN architecture, originally
# published in "End to End Learning for Self-Driving Cars" (Bojarski et al.,
# 2016). The network takes a preprocessed front-camera image as input and
# directly outputs a single steering angle — no hand-crafted feature
# extraction or lane detection pipeline is needed.
#
# Architecture summary (input -> output):
#   Lambda normalisation
#   Conv2D 24 filters, 5x5, stride 2  -> 31x98 feature map
#   Conv2D 36 filters, 5x5, stride 2  -> 14x47 feature map
#   Conv2D 48 filters, 5x5, stride 2  ->  5x22 feature map
#   Conv2D 64 filters, 3x3, stride 1  ->  3x20 feature map
#   Conv2D 64 filters, 3x3, stride 1  ->  1x18 feature map
#   Flatten                            -> 1164 neurons
#   Dropout(0.5)
#   Dense(100) -> Dense(50) -> Dense(10) -> Dense(1)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Why ELU instead of ReLU, Sigmoid, or Softmax?
# ---------------------------------------------------------------------------
# ELU (Exponential Linear Unit) was chosen for all hidden layers for these
# reasons:
#
# vs. ReLU:
#   ReLU outputs 0 for any negative input, which means neurons that fire 0
#   stop receiving gradients and can "die" permanently (the "dying ReLU"
#   problem). ELU outputs a small negative value for negative inputs
#   (e(-x) - 1), keeping gradients alive and allowing the network to push
#   mean activations closer to zero — similar to batch normalisation but
#   without the extra layer. This leads to faster, more stable convergence.
#
# vs. Sigmoid:
#   Sigmoid squashes outputs to (0, 1) and saturates at both ends, causing
#   vanishing gradients in deep networks. It is also not zero-centred, which
#   slows down gradient descent. ELU avoids both of these problems.
#
# vs. Softmax:
#   Softmax converts a vector of values into a probability distribution that
#   sums to 1. It is only appropriate for multi-class classification output
#   layers. Our output is a single continuous steering angle (regression),
#   so Softmax is completely unsuitable here.
#
# The final output layer (Dense(1)) uses NO activation at all, which means
# it performs a linear combination of its inputs. This is correct for a
# regression task where the target (steering angle) can be any real number
# in the range [-1, 1] — we do not want to artificially constrain the output.
# ---------------------------------------------------------------------------


def build_model():
    """
    Build and compile the NVIDIA End-to-End CNN for steering angle regression.

    Returns:
        model : compiled Keras Sequential model ready for training
    """
    model = Sequential([

        # ── Input normalisation ──────────────────────────────────────────
        # Scales pixel values from [0, 255] to [0.0, 1.0] inside the graph.
        # Doing normalisation here (rather than in preprocessing) means it
        # happens on the GPU automatically during training and inference,
        # and is baked into the saved model so TestSimulation.py does not
        # need a separate normalisation step.
        # input_shape=(66, 200, 3) matches the output of preprocess() in
        # preprocessing.py (height=66, width=200, channels=3 YUV).
        Lambda(lambda x: x / 255.0, input_shape=(66, 200, 3)),

        # ── Convolutional feature extraction (NVIDIA architecture) ────────
        # The first three layers use 5x5 kernels with stride 2, which
        # simultaneously extracts features AND downsamples the spatial
        # dimensions (similar to a Conv + MaxPool combination but in one op).
        # The last two layers use 3x3 kernels with stride 1 to capture finer
        # spatial features in the already-reduced feature maps.
        Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),

        # ── Flatten ──────────────────────────────────────────────────────
        # Converts the 3-D feature maps (height x width x filters) into a
        # single 1-D vector of 1164 values so the Dense layers can process them.
        Flatten(),

        # ── Dropout (regularisation) ─────────────────────────────────────
        # Dropout(0.5) randomly sets 50% of the flattened neurons to zero
        # during each training step. This forces the network to learn
        # redundant representations and prevents it from memorising specific
        # training images (overfitting). The effect is that the model
        # generalises better to road conditions not seen during training.
        # Dropout is ONLY active during training; at inference time (when
        # TestSimulation.py runs) Keras automatically disables it so all
        # neurons contribute to the prediction.
        Dropout(0.5),

        # ── Fully connected regression head ──────────────────────────────
        # Three dense layers progressively compress the feature vector
        # (1164 -> 100 -> 50 -> 10) before the final single-neuron output.
        # ELU is used here for the same reasons as in the conv layers.
        Dense(100, activation='elu'),
        Dense(50,  activation='elu'),
        Dense(10,  activation='elu'),

        # ── Output layer ─────────────────────────────────────────────────
        # A single neuron with NO activation (linear output) predicts the
        # continuous steering angle. A linear output is correct for regression
        # — adding an activation like sigmoid or tanh would unnecessarily
        # restrict the range of predictions.
        Dense(1)
    ])

    # ── Compile the model ────────────────────────────────────────────────────
    # optimizer=Adam(lr=1e-3):
    #   Adam (Adaptive Moment Estimation) is chosen because it adapts the
    #   learning rate individually for each parameter using estimates of
    #   first and second moments of the gradients. This makes it much less
    #   sensitive to the initial learning rate than plain SGD and typically
    #   converges faster on image regression tasks.
    #   lr=1e-3 (0.001) is Adam's recommended default starting point.
    #
    # loss='mse':
    #   Mean Squared Error is the standard loss function for regression.
    #   It penalises large steering errors more heavily than small ones
    #   (due to the squaring), encouraging the model to avoid dangerous
    #   large deviations even at the cost of tolerating small ones.
    #   MSE would not be appropriate for a classification task (where
    #   cross-entropy is the correct choice), but here our target is a
    #   continuous value so MSE is correct.
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

    model.summary()
    return model
