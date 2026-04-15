# Self-Driving Car Simulation

**CVI620 — Computer Vision | Final Project**

A CNN-based autonomous driving system trained on the Udacity simulator. The model learns to predict steering angles directly from front-facing camera images using an end-to-end deep learning approach based on the NVIDIA self-driving car architecture.

---

## Team Members

| Member | Responsibilities |
|--------|-----------------|
| Member 1 | Data collection, preprocessing pipeline, augmentation, batch generator |
| Member 2 | Model architecture, training, autonomous testing, README |

---

## Repository Structure

```
self-driving-car/
├── scripts/
│   ├── dataexplore.py        # Steering angle histogram & dataset balancing
│   ├── augmentation.py       # Runtime image augmentation functions
│   ├── preprocessing.py      # Image preprocessing pipeline (crop → YUV → resize)
│   ├── batch_generator.py    # Keras-compatible data generator
│   ├── model.py              # NVIDIA CNN architecture definition
│   ├── train.py              # Model training script with loss curve output
│   └── TestSimulation.py     # Flask-SocketIO server for autonomous driving
├── data/
│   └── driving_log.csv       # Collected driving log (IMG/ excluded — too large)
├── docs/
│   ├── approach.md           # Design decisions and methodology
│   ├── steering_histogram.png
│   └── loss_curve.png
├── requirements.txt
├── model.h5                  # Trained model weights (generated after training)
└── README.md
```

---

## Prerequisites

- Python 3.8
- `virtualenv`
- Udacity Self-Driving Car Simulator — [download here](https://github.com/udacity/self-driving-car-sim/releases) (Term 1)

---

## Setup

**1. Clone the repository**
```bash
git clone <repo-url>
cd self-driving-car
```

**2. Create and activate a virtual environment**
```bash
# macOS / Linux
virtualenv venv --python=python3.8
source venv/bin/activate

# Windows
python -m virtualenv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

> **M1/M2 Mac users:** Replace `tensorflow==2.5.0` in `requirements.txt` with:
> ```
> tensorflow-macos==2.9.0
> tensorflow-metal==0.5.0
> ```

> **NVIDIA GPU users (Windows):** Additionally run:
> ```bash
> pip install tensorflow-gpu==2.5.0
> ```

---

## Data Collection

1. Launch the Udacity simulator
2. Set resolution to `640×480`, quality to `Fastest`, enable windowed mode
3. Select **Training Mode → Track 1**
4. Click **Record** and point the output to the `data/` folder
5. Drive **5 laps forward** then **5 laps in reverse** for balanced coverage
6. Click Record again to stop — `data/IMG/` and `data/driving_log.csv` will be generated

---

## Data Exploration & Balancing

Inspect the raw steering angle distribution and balance overrepresented bins:

```bash
python scripts/dataexplore.py
```

This saves a histogram to `docs/steering_histogram.png` and reports sample counts before and after balancing (capped at 1,000 samples per bin across 25 bins).

---

## Training

```bash
python scripts/train.py
```

The script will:
- Load and balance the driving log
- Split data 80/20 into training and validation sets
- Apply augmentation (flip, brightness, zoom, pan, rotation) during training
- Train the NVIDIA CNN for 30 epochs using the Adam optimizer (MSE loss)
- Save the loss curve to `docs/loss_curve.png`
- Save the trained model as `model.h5`

---

## Model Architecture

Based on the [NVIDIA end-to-end self-driving CNN](https://arxiv.org/abs/1604.07316), adapted with dropout regularization:

| Layer | Details |
|-------|---------|
| Normalization | Lambda (÷ 255) — input: 66×200×3 |
| Conv2D | 24 filters, 5×5, stride 2, ELU |
| Conv2D | 36 filters, 5×5, stride 2, ELU |
| Conv2D | 48 filters, 5×5, stride 2, ELU |
| Conv2D | 64 filters, 3×3, ELU |
| Conv2D | 64 filters, 3×3, ELU |
| Flatten | — |
| Dropout | 50% |
| Dense | 100 units, ELU |
| Dense | 50 units, ELU |
| Dense | 10 units, ELU |
| Dense | 1 unit (steering angle output) |

**Optimizer:** Adam (lr=1e-3) · **Loss:** Mean Squared Error

---

## Preprocessing Pipeline

Every image passes through the following steps before being fed to the model:

1. **Crop** — rows 60–135 (removes sky and hood, keeps road)
2. **Color space** — RGB → YUV (matches NVIDIA paper)
3. **Gaussian blur** — 3×3 kernel to reduce noise
4. **Resize** — 200×66 pixels
5. **Normalize** — pixel values ÷ 255

---

## Autonomous Testing

Ensure `model.h5` is in the project root, then open two terminals with the venv active:

**Terminal 1 — start the prediction server:**
```bash
python scripts/TestSimulation.py
```
You should see `Setting Up ...` — the server is now listening.

**Terminal 2 — launch the simulator:**
1. Open the simulator app
2. Click **Play → Autonomous Mode**

The car will begin steering autonomously. Terminal 1 will stream live `throttle / steering / speed` values.

---

## Approach

please see: docs/approach.md

---

## Challenges

please see: docs/approach.md

---

## Deliverables Checklist

| File | Owner | Status |
|------|-------|--------|
| `dataexplore.py` | Member 1 | ✅ |
| `augmentation.py` | Member 1 | ✅ |
| `preprocessing.py` | Member 1 | ✅ |
| `batch_generator.py` | Member 1 | ✅ |
| `driving_log.csv` | Member 1 | ✅ |
| `steering_histogram.png` | Member 1 | ✅ |
| `model.py` | Member 2 | ✅ |
| `train.py` | Member 2 | ✅ |
| `TestSimulation.py` | Both | ✅ |
| `loss_curve.png` | Member 2 | ✅ |
| `model.h5` | Member 2 | ✅ |
| `README.md` | Member 2 | ✅ |
| `requirements.txt` | Member 1 | ✅ |
| Screen recording (`demo.mp4`) | Member 2 | ⬜ |
| Both names in git history | Both | ⬜ |

---

# Video Submission:
https://youtu.be/_lU_HQS-bes

## References

- Bojarski, M. et al. (2016). *End to End Learning for Self-Driving Cars.* NVIDIA. [arXiv:1604.07316](https://arxiv.org/abs/1604.07316)
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
