import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from dataexplore import load_data, balance_data
from batch_generator import batch_generator
from model import build_model

# ---------------------------------------------------------------------------
# train.py
# ---------------------------------------------------------------------------
# Entry point for training the NVIDIA End-to-End CNN.
#
# Execution order:
#   1. Load and balance the driving log CSV.
#   2. Split into training (80%) and validation (20%) sets.
#   3. Build the model (defined in model.py).
#   4. Create training and validation batch generators.
#   5. Train using model.fit() with the generators.
#   6. Plot and save the loss curves.
#   7. Save the trained model as model.h5.
#
# Run from the project root:
#   python scripts/train.py
# ---------------------------------------------------------------------------

# ── Hyperparameter configuration ─────────────────────────────────────────────
# Centralising these at the top makes it easy to tune without hunting through
# the code.
DATA_PATH  = 'data'   # Root directory containing driving_log.csv and /IMG/
BATCH_SIZE = 32       # Number of images per training step
EPOCHS     = 30       # Full passes through the training data
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── 1. Load and balance the dataset ──────────────────────────────────────
    # load_data() reads driving_log.csv and returns a DataFrame.
    # balance_data() caps over-represented steering bins (near 0) at 1000
    # samples so the model does not learn to always steer straight.
    df = load_data(DATA_PATH + '/driving_log.csv')
    df = balance_data(df, display=False)  # display=False skips the histogram plot

    # ── 2. Train / Validation split ───────────────────────────────────────────
    # We split the centre-camera image paths and their corresponding steering
    # angles into 80% training and 20% validation sets.
    # random_state=42 makes the split reproducible across runs.
    # The validation set is never augmented (controlled inside batch_generator)
    # so it reflects real model performance on unmodified data.
    X_train, X_val, y_train, y_val = train_test_split(
        df['center'].values,     # Array of image file paths
        df['steering'].values,   # Array of corresponding steering angles
        test_size=0.2,
        random_state=42
    )
    print(f'Train: {len(X_train)} | Val: {len(X_val)}')

    # ── 3. Build the model ────────────────────────────────────────────────────
    # Constructs the NVIDIA CNN and compiles it with Adam + MSE.
    # See model.py for architecture details and the reasoning behind each choice.
    model = build_model()

    # ── 4. Create batch generators ────────────────────────────────────────────
    # Generators load and preprocess images on-the-fly in batches, avoiding
    # the need to hold the entire dataset in RAM simultaneously.
    # Training generator applies random augmentation; validation generator does not.
    train_gen = batch_generator(DATA_PATH, X_train, y_train,
                                BATCH_SIZE, is_training=True)
    val_gen   = batch_generator(DATA_PATH, X_val, y_val,
                                BATCH_SIZE, is_training=False)

    # ── 5. Train the model ────────────────────────────────────────────────────
    # steps_per_epoch  : how many batches constitute one epoch for training.
    #                    Integer division ensures we use whole batches only.
    # validation_steps : same concept but for the validation generator.
    # verbose=1        : prints a progress bar + loss values after each epoch.
    # The returned 'history' object contains loss values for every epoch,
    # which we use below to plot the learning curves.
    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=len(X_val) // BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )

    # ── 6. Plot training vs. validation loss ──────────────────────────────────
    # A healthy training run shows both curves decreasing and converging.
    # If training loss keeps falling but validation loss plateaus or rises,
    # the model is overfitting — consider adding more Dropout or reducing EPOCHS.
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'],     label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('docs/loss_curve.png')
    plt.show()
    print('Loss curve saved to docs/loss_curve.png')

    # ── 7. Save the trained model ─────────────────────────────────────────────
    # Saves the full model (architecture + weights + optimiser state) in the
    # legacy HDF5 format (.h5) so that TestSimulation.py can reload it with
    # tf.keras.models.load_model('model.h5').
    model.save('model.h5')
    print('Model saved as model.h5')


if __name__ == '__main__':
    main()
