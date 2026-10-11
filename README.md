\# Facial Expression Recognition — ROB311



Comparison of three facial-expression recognition models on FER-2013.



\## Models



1\. LBP + KNN

2\. HOG + Linear SVM

3\. mini-Xception CNN



\## Application



The Gradio interface accepts:



\- an uploaded image

\- a webcam image



The application:



1\. detects the largest face using OpenCV Haar Cascade,

2\. crops the same face for all three models,

3\. runs each model,

4\. displays:

&#x20;  - predicted emotion,

&#x20;  - confidence,

&#x20;  - inference latency.



Face detection latency is measured separately.



\## Emotion classes



\- Angry

\- Disgust

\- Fear

\- Happy

\- Sad

\- Surprise

\- Neutral



\## Installation



Python 3.11 is recommended.



```bash

python -m venv .venv

