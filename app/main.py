import time
import gradio as gr

from face_utils import detect_largest_face
from model_wrappers import (
    predict_model1,
    predict_model2,
    predict_model3,
)


# =========================================================
# RUN ALL MODELS
# =========================================================

def run_all_models(image):

    if image is None:
        return (
            None,
            "No image provided.",
            "", "", "",
            "", "", "",
            "", "", ""
        )

    # -----------------------------------------------------
    # Face detection
    # -----------------------------------------------------

    detection_start = time.perf_counter()

    face_crop, annotated_image, bbox = detect_largest_face(image)

    detection_latency_ms = (
        time.perf_counter() - detection_start
    ) * 1000

    if face_crop is None:
        return (
            image,
            f"No face detected. Detection latency: "
            f"{detection_latency_ms:.2f} ms",

            "", "", "",
            "", "", "",
            "", "", ""
        )

    # -----------------------------------------------------
    # Run the 3 models
    # -----------------------------------------------------

    result1 = predict_model1(face_crop)
    result2 = predict_model2(face_crop)
    result3 = predict_model3(face_crop)

    # -----------------------------------------------------
    # Format outputs
    # -----------------------------------------------------

    model1_label = result1["label"]
    model1_conf = f"{result1['confidence'] * 100:.2f}%"
    model1_latency = f"{result1['latency_ms']:.2f} ms"

    model2_label = result2["label"]
    model2_conf = f"{result2['confidence'] * 100:.2f}%"
    model2_latency = f"{result2['latency_ms']:.2f} ms"

    model3_label = result3["label"]
    model3_conf = f"{result3['confidence'] * 100:.2f}%"
    model3_latency = f"{result3['latency_ms']:.2f} ms"

    bbox_text = (
        f"Face detected at x={bbox[0]}, y={bbox[1]}, "
        f"w={bbox[2]}, h={bbox[3]} | "
        f"Detection latency: {detection_latency_ms:.2f} ms"
    )

    return (
        annotated_image,
        bbox_text,

        model1_label,
        model1_conf,
        model1_latency,

        model2_label,
        model2_conf,
        model2_latency,

        model3_label,
        model3_conf,
        model3_latency,
    )


# =========================================================
# GRADIO INTERFACE
# =========================================================

with gr.Blocks(
    title="Facial Expression Recognition Comparison"
) as demo:

    gr.Markdown(
        """
        # Facial Expression Recognition

        Compare three facial-expression recognition models
        on the **same detected face**.

        ### Models
        - **Model 1:** LBP + KNN
        - **Model 2:** HOG + Linear SVM
        - **Model 3:** mini-Xception CNN

        Use either:
        - an uploaded image for manual prediction
        - the webcam for real-time prediction
        """
    )

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    gr.Markdown("## Input")

    with gr.Row():

        upload_image = gr.Image(
            label="Upload image",
            sources=["upload"],
            type="numpy"
        )

        webcam_image = gr.Image(
            label="Live webcam",
            sources=["webcam"],
            type="numpy",
            streaming=True
        )

    run_button = gr.Button(
        "Run uploaded image",
        variant="primary"
    )

    # -----------------------------------------------------
    # DETECTED FACE
    # -----------------------------------------------------

    gr.Markdown("## Face detection")

    detected_image = gr.Image(
        label="Detected face",
        type="numpy"
    )

    detect_info = gr.Textbox(
        label="Face detection information",
        interactive=False
    )

    # -----------------------------------------------------
    # MODEL RESULTS
    # -----------------------------------------------------

    gr.Markdown("## Model comparison")

    with gr.Row():

        with gr.Column():

            gr.Markdown("### Model 1 — LBP + KNN")

            model1_label = gr.Textbox(
                label="Emotion",
                interactive=False
            )

            model1_conf = gr.Textbox(
                label="Confidence",
                interactive=False
            )

            model1_latency = gr.Textbox(
                label="Latency",
                interactive=False
            )

        with gr.Column():

            gr.Markdown("### Model 2 — HOG + SVM")

            model2_label = gr.Textbox(
                label="Emotion",
                interactive=False
            )

            model2_conf = gr.Textbox(
                label="Confidence",
                interactive=False
            )

            model2_latency = gr.Textbox(
                label="Latency",
                interactive=False
            )

        with gr.Column():

            gr.Markdown("### Model 3 — mini-Xception")

            model3_label = gr.Textbox(
                label="Emotion",
                interactive=False
            )

            model3_conf = gr.Textbox(
                label="Confidence",
                interactive=False
            )

            model3_latency = gr.Textbox(
                label="Latency",
                interactive=False
            )

    outputs = [
        detected_image,
        detect_info,

        model1_label,
        model1_conf,
        model1_latency,

        model2_label,
        model2_conf,
        model2_latency,

        model3_label,
        model3_conf,
        model3_latency,
    ]

    # -----------------------------------------------------
    # MANUAL UPLOAD
    # -----------------------------------------------------

    run_button.click(
        fn=run_all_models,
        inputs=upload_image,
        outputs=outputs,
    )

    # -----------------------------------------------------
    # REAL-TIME WEBCAM
    # -----------------------------------------------------

    webcam_image.stream(
        fn=run_all_models,
        inputs=webcam_image,
        outputs=outputs,
        stream_every=0.5
    )


# =========================================================
# START APP
# =========================================================

if __name__ == "__main__":
    demo.launch()