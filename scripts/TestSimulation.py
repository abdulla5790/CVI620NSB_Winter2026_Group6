import os

# Suppress TensorFlow's verbose startup logs (INFO and WARNING messages).
# Must be set before importing TensorFlow/Keras.
# Log levels: 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
print('Setting Up ...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import socketio
import eventlet
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2

# ---------------------------------------------------------------------------
# TestSimulation.py
# ---------------------------------------------------------------------------
# This script acts as the autonomous driving server that connects the trained
# CNN model to the Udacity simulator in real time.
#
# How it works:
#   1. The Udacity simulator (run in "Autonomous Mode") connects to this
#      server via a WebSocket on port 4567.
#   2. On every simulation frame the simulator emits a 'telemetry' event
#      containing the current speed and a base64-encoded front-camera image.
#   3. This server decodes the image, applies the same preprocessing used
#      during training, and feeds it to the loaded model.
#   4. The model predicts a steering angle; throttle is computed from speed.
#   5. Both values are sent back to the simulator via the 'steer' event,
#      which the simulator applies to the virtual car.
#
# Usage:
#   1. Activate the same conda environment used for training (cvi620).
#   2. Run: python TestSimulation.py
#   3. Open the simulator -> select a track -> click "Autonomous Mode".
# ---------------------------------------------------------------------------

# ── WebSocket server setup ───────────────────────────────────────────────────
# socketio.Server() creates a Socket.IO server instance that handles the
# real-time bidirectional communication with the Udacity simulator.
sio = socketio.Server()

# Flask is used as the underlying HTTP/WSGI application. Socket.IO wraps it
# later (via socketio.Middleware) to add WebSocket support on top of HTTP.
app = Flask(__name__)

# Maximum speed cap in mph. The throttle formula (1.0 - speed/maxSpeed)
# produces throttle=1.0 when stationary and throttle=0.0 at maxSpeed,
# effectively acting as a simple proportional speed controller.
# Increase this value to allow the car to drive faster (may reduce stability).
maxSpeed = 10


def preProcessing(img):
    """
    Apply the same preprocessing pipeline used during training to a live
    simulator frame.

    It is critical that this function is IDENTICAL to preprocess() in
    preprocessing.py. Any difference (e.g. a different crop range or missing
    blur step) would mean the model receives input it was never trained on,
    causing unpredictable steering behaviour.

    Steps:
        1. Crop rows 60-134: removes sky and car hood, keeps road surface.
        2. Convert RGB -> YUV: matches the colour space the model was trained on.
        3. Gaussian blur (3x3): reduces high-frequency noise.
        4. Resize to 200x66: matches the CNN's expected input_shape=(66,200,3).
        5. Normalise to [0,1]: scales pixel values for the neural network.

    Args:
        img : numpy array (H, W, 3), RGB, uint8 — decoded from the simulator frame

    Returns:
        img : numpy array (66, 200, 3), float64 in [0.0, 1.0]
    """
    img = img[60:135, :, :]                         # 1. Crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)      # 2. RGB -> YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)          # 3. Gaussian blur
    img = cv2.resize(img, (200, 66))                # 4. Resize (width, height)
    # img = img / 255                                 # 5. Normalise to [0, 1]
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    """
    Called automatically on every simulation frame when the simulator sends
    telemetry data over the WebSocket.

    Args:
        sid  : Socket.IO session ID of the connected simulator (managed internally)
        data : dict containing:
                 'speed' : current vehicle speed as a string (mph)
                 'image' : base64-encoded JPEG of the front-camera frame
    """
    # Parse current speed from the telemetry payload
    speed = float(data['speed'])

    # Decode the base64 image string -> JPEG bytes -> PIL Image -> numpy array
    # The simulator encodes each frame as a base64 JPEG to transmit it over
    # the WebSocket as a plain text string.
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)  # Convert PIL Image to (H, W, 3) numpy array

    # Apply the same preprocessing used during training
    image = preProcessing(image)

    # Add a batch dimension: model.predict() expects shape (batch, H, W, C)
    # np.array([image]) converts (66, 200, 3) -> (1, 66, 200, 3)
    image = np.array([image])

    # Run the preprocessed frame through the CNN to predict the steering angle.
    # The output is a (1, 1) array; float() extracts the scalar value.
    steering = float(model.predict(image))

    # Compute throttle using a simple proportional controller:
    #   - At speed=0:        throttle = 1.0  (full acceleration)
    #   - At speed=maxSpeed: throttle = 0.0  (coast / no acceleration)
    #   - Above maxSpeed:    throttle < 0.0  (slight braking effect)
    throttle = 1.0 - speed / maxSpeed

    print(f'throttle={throttle:.3f}, steering={steering:.4f}, speed={speed:.2f}')

    # Send the computed steering and throttle back to the simulator
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    """
    Called once when the Udacity simulator first establishes a WebSocket
    connection with this server.

    We immediately send a neutral control command (steering=0, throttle=0)
    to acknowledge the connection and prevent the simulator from waiting
    indefinitely for its first command before starting the physics simulation.
    """
    print('Connected')
    sendControl(0, 0)


def sendControl(steering, throttle):
    """
    Emit a 'steer' Socket.IO event back to the simulator with the computed
    steering angle and throttle value.

    The simulator listens for this specific event name and applies the values
    directly to the car's steering and acceleration actuators.

    Both values must be sent as strings (.__str__()) because the simulator's
    Socket.IO client expects string-typed fields in the JSON payload.

    Args:
        steering : float in [-1, 1] — negative=left, positive=right
        throttle : float in [0, 1] — 0=no throttle, 1=full throttle
    """
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == "__main__":
    # Load the trained model from disk.
    # The .h5 file contains the full model: architecture, weights, and
    # compile settings. It must exist in the same directory as this script
    # (or provide the full path).
    model = load_model('model.h5')

    # Wrap the Flask app with Socket.IO middleware so that WebSocket
    # upgrade requests from the simulator are handled by socketio.Server
    # while regular HTTP requests fall through to Flask.
    app = socketio.Middleware(sio, app)

    # Start the eventlet WSGI server on port 4567.
    # eventlet is an async networking library that allows the server to handle
    # the continuous stream of telemetry events from the simulator efficiently.
    # Port 4567 is hardcoded in the Udacity simulator and cannot be changed.
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
