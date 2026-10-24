# Smart Facial Recognition System

Real-time face recognition with a webcam GUI.
Pipeline: **face detection (Haar) -> crop -> FaceNet embedding (512-d) -> cosine matching -> Access Granted / Denied**.

> Current model: pre-trained FaceNet (`keras-facenet`) used as a feature extractor (transfer learning, no extra training).

## Project structure

| File | Role |
|---|---|
| `detector.py` | Face detection (Haar Cascade) |
| `embedder.py` | Crop + preprocessing + FaceNet embedding (shared by app and evaluation) |
| `matcher.py` | Cosine distance matching and the Unknown threshold |
| `database.py` | Saving / loading enrolled embeddings and face images |
| `gui.py` | Tkinter camera interface |
| `evaluate.py` | Evaluation with the same pipeline (Accuracy, Precision, Recall, ROC/EER) |
| `main.py` | Entry point |

## Setup

```
pip install -r requirements.txt
```

## Run the app

```
python main.py
```

Type a name and press **Save the new person** while your face is detected.
Known faces get a green box (`Access Granted`), unknown faces a red box (`Access Denied`).

## Evaluation

Put images in one folder per person:

```
eval_data/
    person_a/  1.jpg 2.jpg ...
    person_b/  ...
```

```
python evaluate.py --data eval_data
python evaluate.py --data eval_data --no-detect   # if images are already cropped
```

Images are split 80% / 10% / 10% (templates / threshold tuning / final test).
Some people are held out as "Unknown" to test rejection. Output: Overall Accuracy,
per-person Precision/Recall, EER, AUC, and `roc.png`.

## Privacy

`face_embeddings.pkl`, `dataset/` and evaluation images contain biometric data of real people.
They are excluded by `.gitignore` and must never be committed.
Only enroll people who gave their consent.
