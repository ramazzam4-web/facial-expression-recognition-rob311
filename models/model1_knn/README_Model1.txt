MODEL 1 — LBP + KNN
=====================

Files to share
--------------
1. model1_lbp_knn.pkl
2. Model1_LBP_KNN.ipynb
3. README_Model1.txt

Model configuration
-------------------
LBP method        : nri_uniform
P                 : 8
R                 : 1
Spatial grid      : 6 x 6
Normalization     : Hellinger
Feature dimension : 2124

KNN
---
K                 : 9
Weights           : distance
Metric            : Euclidean

FER-2013 results
----------------
Test accuracy     : 50.68%
Macro-F1          : 0.4825
Mean latency      : 79.75 ms
Median latency    : 78.89 ms

Class order
-----------
['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


REQUIRED PACKAGES
=================
pip install numpy pillow scikit-image scikit-learn==1.6.1 joblib


HOW TO LOAD AND USE MODEL 1
===========================

import time
import joblib
import numpy as np

from PIL import Image
from skimage.feature import local_binary_pattern


# ------------------------------------------------------------
# 1. Load model bundle
# ------------------------------------------------------------

bundle = joblib.load("model1_lbp_knn.pkl")

knn = bundle["knn"]
classes = bundle["classes"]
config = bundle["config"]

print("Model 1 loaded")
print("Classes:", classes)
print("Configuration:", config)


# ------------------------------------------------------------
# 2. LBP feature extraction
# ------------------------------------------------------------

def lbp_num_bins(P, method):
    if method == "default":
        return 2 ** P
    if method == "uniform":
        return P + 2
    if method == "nri_uniform":
        return P * (P - 1) + 3
    raise ValueError(f"Unsupported LBP method: {method}")


def extract_model1_lbp_features(image):
    image = np.asarray(image, dtype=np.uint8)

    P = config["P"]
    R = config["R"]
    grid = config["grid"]
    method = config["lbp_method"]

    lbp = local_binary_pattern(
        image,
        P=P,
        R=R,
        method=method
    )

    n_bins = lbp_num_bins(P, method)

    height, width = lbp.shape
    block_h = height // grid
    block_w = width // grid

    features = []

    for row in range(grid):
        for col in range(grid):
            y1 = row * block_h
            y2 = (row + 1) * block_h
            x1 = col * block_w
            x2 = (col + 1) * block_w

            block = lbp[y1:y2, x1:x2]

            hist = np.bincount(
                block.astype(np.int64).ravel(),
                minlength=n_bins
            ).astype(np.float32)[:n_bins]

            # Hellinger normalization
            hist /= (hist.sum() + 1e-12)
            hist = np.sqrt(hist)

            features.append(hist)

    feature_vector = np.concatenate(features).astype(np.float32)

    if feature_vector.shape[0] != config["feature_dimension"]:
        raise ValueError(
            f"Unexpected feature size: {feature_vector.shape[0]}"
        )

    return feature_vector


# ------------------------------------------------------------
# 3. Prediction function
# ------------------------------------------------------------

def predict_model1(image):
    # image can be a PIL Image or a NumPy image array.

    start = time.perf_counter()

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # FER-2013 preprocessing
    image = image.convert("L").resize((48, 48))
    image_array = np.array(image, dtype=np.uint8)

    features = extract_model1_lbp_features(
        image_array
    ).reshape(1, -1)

    # One KNN search
    probabilities = knn.predict_proba(features)

    best_position = int(np.argmax(probabilities[0]))
    predicted_label = int(knn.classes_[best_position])

    emotion = classes[predicted_label]
    confidence = float(probabilities[0, best_position])

    latency_ms = (time.perf_counter() - start) * 1000

    return emotion, confidence, latency_ms


# ------------------------------------------------------------
# 4. Example
# ------------------------------------------------------------

image = Image.open("example_face.jpg")

emotion, confidence, latency = predict_model1(image)

print("Emotion:", emotion)
print(f"Confidence: {confidence * 100:.2f}%")
print(f"Latency: {latency:.2f} ms")


IMPORTANT FOR THE INTERFACE
===========================
The input to Model 1 should be the detected/cropped FACE image, not the full
camera frame.

Recommended pipeline:

Webcam/image
    -> face detection
    -> crop face
    -> predict_model1(face_crop)
    -> display:
         emotion
         confidence
         latency

The prediction function already converts the face crop to grayscale and
resizes it to 48 x 48.

Do not change the preprocessing parameters unless the model is retrained.
