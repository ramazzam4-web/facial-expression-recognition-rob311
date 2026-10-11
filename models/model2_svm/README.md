# HOG + Linear SVM (one-vs-rest) — FER-2013

| Fichier | Rôle |
|---|---|
| `hog_svm.py` | Module importé par l'app : extraction HOG + prédiction |
| `train_hog_svm.py` | Entraînement sur FER-2013 → `hog_svm_model.pkl` + `confusion_matrix_hog_svm.png` |
| `hog_svm_model.pkl` | Modèle entraîné (généré par le script, à mettre à côté de `hog_svm.py`) |
| `requirements.txt` | Dépendances |

## Entraîner (une seule fois, ~5-7 min)

```bash
pip install -r requirements.txt
python train_hog_svm.py chemin/vers/fer2013.csv     # version CSV (emotion, pixels, Usage)
python train_hog_svm.py chemin/vers/fer2013/        # version images (dossiers train/ et test/)
```

Le script affiche la précision sur le test, le rapport par classe et la latence moyenne par visage. Un `ConvergenceWarning` pendant le choix de C (grands C) est sans gravité.

Test rapide sur une photo (détection Haar) : `python hog_svm.py photo.jpg`

## Utilisation dans l'app

```python
from hog_svm import HogSvmClassifier

hog_svm = HogSvmClassifier()        # au démarrage de l'app, une seule fois
res = hog_svm.predict(face_crop)    # à chaque frame
# {'label': 'Happy', 'confidence': 0.71, 'latency_ms': 1.6,
#  'probs': {'Angry': 0.05, 'Disgust': 0.01, ..., 'Neutral': 0.08}}
```

- **Entrée** : le crop du visage (sortie de Haar), en niveaux de gris `(H, W)` ou RGB `(H, W, 3)`, n'importe quelle taille. Redimensionné en 48×48 en interne.
- **Sortie** : `label` (str), `confidence` (0-1), `latency_ms`, `probs` (une proba par émotion, somme = 1). `probs` se branche directement dans un `gr.Label` Gradio.

## Points d'attention

1. **OpenCV < 5.** `pip install opencv-python` installe maintenant OpenCV 5, qui n'a plus `cv2.CascadeClassifier` ni les fichiers Haar. Installer `"opencv-python<5"` (déjà dans `requirements.txt`).
2. **BGR.** Une frame `cv2.VideoCapture` est en BGR : la convertir en gris (`cv2.COLOR_BGR2GRAY`, nécessaire pour Haar de toute façon) et passer le même crop gris aux 3 modèles. Ne pas passer un crop BGR couleur.
3. **Latence.** `latency_ms` = HOG + SVM, sans la détection de visage (commune aux 3 modèles). Mesurer de la même façon pour KNN et CNN pour que le tableau soit comparable. Le premier appel est plus lent (~8 ms contre ~1-2 ms ensuite) : faire un appel à blanc au démarrage ou ignorer la première frame.
4. **Versions.** Le `.pkl` dépend des versions de scikit-learn et numpy de la machine qui l'a créé (elles sont écrites dans le pickle, clé `versions`). Si le chargement plante ou affiche `InconsistentVersionWarning`, relancer `train_hog_svm.py` dans l'environnement de l'app.

## Choix techniques (pour le rapport)

- **HOG** : 9 orientations, cellules 8×8 px, blocs 2×2 cellules, normalisation L2-Hys → 900 features par visage 48×48.
- **Classifieur** : `StandardScaler` → `LinearSVC`, qui entraîne 7 SVM « une émotion vs le reste » ; le score le plus haut gagne.
- **C** choisi sur une validation (15 % du train), jamais sur le test.
- **Confiance** : un SVM ne donne pas de probabilité. `CalibratedClassifierCV` (calibration de Platt) transforme les scores en probabilités ; ça ne change quasiment pas la classe prédite.
- **Split** : train = `Training`, test = `PublicTest` + `PrivateTest` (7 178 images, identique au dossier `test/` de la version images).
- **Ordre de grandeur attendu** : ~44 % de précision sur FER-2013 pour HOG + SVM linéaire (arXiv 2607.15288), loin derrière le CNN (60-67 %). Disgust, très peu représenté, sera la classe la plus faible.
