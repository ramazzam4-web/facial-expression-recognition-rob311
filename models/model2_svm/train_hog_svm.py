"""Train the HOG + Linear SVM (one-vs-rest) emotion classifier on FER-2013.

    python train_hog_svm.py path/to/fer2013.csv    # CSV version (emotion, pixels, Usage)
    python train_hog_svm.py path/to/fer2013/       # image version (train/ and test/ folders)

Writes hog_svm_model.pkl (loaded by hog_svm.py) and confusion_matrix_hog_svm.png,
and prints test accuracy and latency per face.
"""
import csv
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from hog_svm import DEFAULT_MODEL, EMOTIONS, HOG_PARAMS, HogSvmClassifier, extract_hog

# Folder names in the image version ("surprise" or "surprised"...): first 3 letters
FOLDER_TO_LABEL = {"ang": 0, "dis": 1, "fea": 2, "hap": 3, "sad": 4, "sur": 5, "neu": 6}


def load_fer_csv(path):
    """Usage 'Training' -> train set, 'PublicTest' + 'PrivateTest' -> test set."""
    imgs_train, y_train, imgs_test, y_test = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            img = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            if row["Usage"] == "Training":
                imgs_train.append(img)
                y_train.append(int(row["emotion"]))
            else:
                imgs_test.append(img)
                y_test.append(int(row["emotion"]))
    return imgs_train, np.array(y_train), imgs_test, np.array(y_test)


def load_fer_folders(root):
    """root/train/<emotion>/*.jpg and root/test/<emotion>/*.jpg"""
    out = []
    for split in ["train", "test"]:
        imgs, labels = [], []
        for class_dir in sorted(p for p in (Path(root) / split).iterdir() if p.is_dir()):
            label = FOLDER_TO_LABEL[class_dir.name.lower()[:3]]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    imgs.append(np.array(Image.open(img_path).convert("L")))
                    labels.append(label)
        out += [imgs, np.array(labels)]
    return out


def main():
    data_path = Path(sys.argv[1])
    print("Loading FER-2013...")
    if data_path.suffix == ".csv":
        imgs_train, y_train, imgs_test, y_test = load_fer_csv(data_path)
    else:
        imgs_train, y_train, imgs_test, y_test = load_fer_folders(data_path)
    print(f"  {len(imgs_train)} train / {len(imgs_test)} test images")

    print("Extracting HOG features...")
    t0 = time.perf_counter()
    X_train = np.array([extract_hog(img) for img in imgs_train])
    X_test = np.array([extract_hog(img) for img in imgs_test])
    print(f"  {X_train.shape[1]} features per face ({time.perf_counter() - t0:.0f} s)")

    # 1) Choose C (regularisation strength) on a validation split of the training set
    print("Choosing C on a validation split...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=0)
    best_C, best_acc = None, -1.0
    for C in [0.001, 0.01, 0.1, 1.0]:
        svm = make_pipeline(StandardScaler(), LinearSVC(C=C, dual=False))
        acc = svm.fit(X_tr, y_tr).score(X_val, y_val)
        print(f"  C={C:<6} validation accuracy = {acc:.3f}")
        if acc > best_acc:
            best_C, best_acc = C, acc

    # 2) Final model on the whole training set.
    #    LinearSVC = 7 one-vs-rest SVMs ("emotion vs rest"), the highest score wins.
    #    CalibratedClassifierCV maps the SVM scores to probabilities (Platt scaling),
    #    which gives the "confidence" shown in the app (an SVM has no probability by itself).
    print(f"Training final model (C={best_C})...")
    model = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(LinearSVC(C=best_C, dual=False), cv=3, ensemble=False),
    )
    model.fit(X_train, y_train)

    # 3) Evaluation on the test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {test_acc:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=EMOTIONS, digits=3))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=EMOTIONS, normalize="true",
        values_format=".2f", cmap="Greys", colorbar=False, xticks_rotation=45)
    plt.title(f"HOG + Linear SVM (OvR) - test accuracy {test_acc:.1%}")
    plt.tight_layout()
    plt.savefig(DEFAULT_MODEL.with_name("confusion_matrix_hog_svm.png"), dpi=150)

    # 4) Save everything the app needs in one pickle
    with open(DEFAULT_MODEL, "wb") as f:
        pickle.dump({
            "model": model,
            "hog_params": HOG_PARAMS,
            "test_accuracy": test_acc,
            "versions": f"scikit-learn {sklearn.__version__}, numpy {np.__version__}",
        }, f)
    print(f"Saved {DEFAULT_MODEL.name}")

    # 5) Latency per face, measured exactly the way the app calls the model
    clf = HogSvmClassifier()
    clf.predict(imgs_test[0])  # warm-up
    times = [clf.predict(img)["latency_ms"] for img in imgs_test[:300]]
    print(f"Latency (HOG + SVM): {np.mean(times):.2f} ms +/- {np.std(times):.2f} ms per face")


if __name__ == "__main__":
    main()
