import sys
import time
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.feature import local_binary_pattern


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL1_DIR = PROJECT_ROOT / "models" / "model1_knn"
MODEL2_DIR = PROJECT_ROOT / "models" / "model2_svm"
MODEL3_DIR = PROJECT_ROOT / "models" / "model3_cnn"


# ---------------------------------------------------------
# MODEL 1 — LBP + KNN
# ---------------------------------------------------------

model1_bundle = joblib.load(MODEL1_DIR / "model1_lbp_knn.pkl")

model1_knn = model1_bundle["knn"]
model1_classes = model1_bundle["classes"]
model1_config = model1_bundle["config"]


def _lbp_num_bins(P, method):
    if method == "default":
        return 2 ** P
    if method == "uniform":
        return P + 2
    if method == "nri_uniform":
        return P * (P - 1) + 3
    raise ValueError(f"Unsupported LBP method: {method}")


def _extract_model1_features(image):
    image = np.asarray(image, dtype=np.uint8)

    P = model1_config["P"]
    R = model1_config["R"]
    grid = model1_config["grid"]
    method = model1_config["lbp_method"]

    lbp = local_binary_pattern(
        image,
        P=P,
        R=R,
        method=method
    )

    n_bins = _lbp_num_bins(P, method)

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

            hist /= (hist.sum() + 1e-12)
            hist = np.sqrt(hist)

            features.append(hist)

    return np.concatenate(features).astype(np.float32)


def predict_model1(face_crop):
    start = time.perf_counter()

    image = Image.fromarray(np.asarray(face_crop, dtype=np.uint8))
    image = image.convert("L").resize((48, 48))

    image_array = np.array(image, dtype=np.uint8)

    features = _extract_model1_features(
        image_array
    ).reshape(1, -1)

    probabilities = model1_knn.predict_proba(features)

    best_position = int(np.argmax(probabilities[0]))
    predicted_label = int(model1_knn.classes_[best_position])

    emotion = model1_classes[predicted_label]
    confidence = float(probabilities[0, best_position])

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "label": emotion.title(),
        "confidence": confidence,
        "latency_ms": latency_ms,
        "probs": None,
    }


# ---------------------------------------------------------
# MODEL 2 — HOG + SVM
# ---------------------------------------------------------

sys.path.insert(0, str(MODEL2_DIR))

from hog_svm import HogSvmClassifier

model2 = HogSvmClassifier(
    MODEL2_DIR / "hog_svm_model.pkl"
)


def predict_model2(face_crop):
    result = model2.predict(face_crop)

    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
        "probs": result["probs"],
    }


# ---------------------------------------------------------
# MODEL 3 — mini-Xception / TensorFlow SavedModel
# ---------------------------------------------------------

model3 = tf.saved_model.load(str(MODEL3_DIR))
model3_predict_fn = model3.signatures["serving_default"]

# TEMPORARY class order.
# We will replace this after your teammate confirms it.
MODEL3_CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def predict_model3(face_crop):
    start = time.perf_counter()

    face = np.asarray(face_crop, dtype=np.float32)

    # Resize to 48x48
    import cv2
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)

    # TEMPORARY normalization.
    # We will update this after your teammate confirms training preprocessing.
    face = face / 255.0

    # Shape: (1, 48, 48, 1)
    face = face[np.newaxis, :, :, np.newaxis]

    output = model3_predict_fn(
        face=tf.constant(face, dtype=tf.float32)
    )

    probs = output["output_0"].numpy()[0]

    best = int(np.argmax(probs))

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "label": MODEL3_CLASSES[best],
        "confidence": float(probs[best]),
        "latency_ms": latency_ms,
        "probs": {
            name: float(prob)
            for name, prob in zip(MODEL3_CLASSES, probs)
        },
    }