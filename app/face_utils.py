import cv2
import numpy as np


# Load OpenCV's frontal-face Haar cascade once
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_largest_face(image):
    """
    Input:
        image: RGB NumPy array from Gradio

    Returns:
        face_crop_gray: cropped face in grayscale
        annotated_rgb: original image with a rectangle around the detected face
        bbox: (x, y, w, h)

    If no face is detected:
        returns None, original image, None
    """

    if image is None:
        return None, None, None

    image = np.asarray(image)

    # Gradio gives RGB images
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )

    if len(faces) == 0:
        return None, image, None

    # Keep the largest detected face
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    face_crop_gray = gray[y:y+h, x:x+w]

    annotated = image.copy()

    cv2.rectangle(
        annotated,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    return face_crop_gray, annotated, (x, y, w, h)