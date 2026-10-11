"""HOG + Linear SVM (one-vs-rest) facial expression classifier.

This is the module the app imports. The model is trained by train_hog_svm.py.

    from hog_svm import HogSvmClassifier

    clf = HogSvmClassifier()             # load once, at app startup
    result = clf.predict(face_crop)      # once per frame
    # {'label': 'Happy', 'confidence': 0.71, 'latency_ms': 1.4,
    #  'probs': {'Angry': 0.05, 'Disgust': 0.01, ..., 'Neutral': 0.08}}

Quick standalone test on a photo (Haar face detection, needs OpenCV 4.x):

    python hog_svm.py photo.jpg
"""
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog

# FER-2013 label order: label i <-> EMOTIONS[i]
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMG_SIZE = 48
DEFAULT_MODEL = Path(__file__).with_name("hog_svm_model.pkl")

# HOG settings, shared by training and inference.
# 48x48 face -> 6x6 cells of 8x8 px -> 5x5 overlapping blocks of 2x2 cells
# -> 5*5 blocks * (2*2 cells * 9 orientations) = 900 features per face
HOG_PARAMS = dict(
    orientations=9,          # gradient direction bins over 0-180 degrees
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",     # L2 normalisation per block (+ clipping): robust to lighting
)


def to_gray_48(face):
    """Face crop (gray or RGB, any size) -> 48x48 grayscale float32 image."""
    face = np.asarray(face)
    if face.ndim == 3:  # colour image, RGB order (what Gradio gives)
        face = face[..., :3] @ np.array([0.299, 0.587, 0.114])
    face = face.astype(np.float32)
    if face.shape != (IMG_SIZE, IMG_SIZE):
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return face


def extract_hog(face):
    """Face crop -> HOG descriptor (1D array of 900 values)."""
    return hog(to_gray_48(face), **HOG_PARAMS)


class HogSvmClassifier:
    def __init__(self, model_path=DEFAULT_MODEL):
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        if saved["hog_params"] != HOG_PARAMS:
            raise ValueError("HOG settings changed since training: rerun train_hog_svm.py")
        self.model = saved["model"]  # sklearn pipeline: StandardScaler -> calibrated LinearSVC
        self.names = [EMOTIONS[c] for c in self.model.classes_]

    def predict(self, face):
        """face: face crop, gray (H, W) or RGB (H, W, 3), any size.

        Returns label, confidence (0-1), latency_ms (HOG + SVM only, face
        detection excluded) and probs (one value per emotion, sums to 1).
        """
        t0 = time.perf_counter()
        features = extract_hog(face).reshape(1, -1)
        probs = self.model.predict_proba(features)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        best = int(np.argmax(probs))
        return {
            "label": self.names[best],
            "confidence": float(probs[best]),
            "latency_ms": latency_ms,
            "probs": {name: float(p) for name, p in zip(self.names, probs)},
        }


if __name__ == "__main__":
    import sys

    from PIL import Image

    gray = np.array(Image.open(sys.argv[1]).convert("L"))
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        sys.exit("No face detected")
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])  # keep the largest face

    result = HogSvmClassifier().predict(gray[y:y + h, x:x + w])
    print(f"{result['label']}  (confidence {result['confidence']:.2f}, {result['latency_ms']:.1f} ms)")
