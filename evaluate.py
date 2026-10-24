"""
evaluate.py - يقيس النظام بنفس pipeline التطبيق (detector -> crop -> embedder -> matcher).

الاستخدام:
    python evaluate.py --data eval_data
    python evaluate.py --data eval_data --no-detect     (إذا كانت الصور مقصوصة مسبقًا)

هيكل البيانات:
    eval_data/
        ali/   1.jpg 2.jpg ...
        sara/  1.jpg 2.jpg ...
"""
import argparse
import os
import random

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

from matcher import FaceMatcher

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_embeddings(root, detector, embedder, detect=True):
    """dict: اسم -> قائمة بصمات. نفس مسار التطبيق: detect -> crop -> embed"""
    data, skipped = {}, 0
    for person in sorted(os.listdir(root)):
        folder = os.path.join(root, person)
        if not os.path.isdir(folder):
            continue
        embs = []
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            img = cv2.imread(os.path.join(folder, fn))
            if img is None:
                skipped += 1
                continue
            if detect:
                boxes = detector.detect(img)
                if len(boxes) == 0:
                    skipped += 1
                    continue
                box = max(boxes, key=lambda b: b[2] * b[3])  # أكبر وجه
                embs.append(embedder.embed_from_frame(img, box))
            else:
                embs.append(embedder.embed(img))
        if embs:
            data[person] = embs
        print(f"  {person}: {len(embs)} images")
    print(f"Skipped {skipped} images (no face detected or unreadable file)")
    return data


def split_person(embs, rng):
    """80% تدريب / 10% تحقق / 10% اختبار"""
    idx = list(range(len(embs)))
    rng.shuffle(idx)
    n = len(idx)
    n_val = max(1, round(n * 0.1))
    n_test = max(1, round(n * 0.1))
    train = [embs[i] for i in idx[: n - n_val - n_test]]
    val = [embs[i] for i in idx[n - n_val - n_test: n - n_test]]
    test = [embs[i] for i in idx[n - n_test:]]
    return train, val, test


def make_template(embs):
    t = np.mean(embs, axis=0)
    return t / np.linalg.norm(t)


def run_identification(probes, db, threshold):
    """نفس FaceMatcher المستخدم في التطبيق"""
    matcher = FaceMatcher(threshold=threshold)
    y_true, y_pred = [], []
    for label, emb in probes:
        identity, _ = matcher.recognize(emb, db)
        y_true.append(label)
        y_pred.append(identity)
    return y_true, y_pred


def verification_scores(probes, db):
    """كل probe مقابل كل شخص مسجّل: label=1 إذا نفس الشخص"""
    dists, labels = [], []
    for label, emb in probes:
        for name, tmpl in db.items():
            dists.append(cosine(emb, tmpl))
            labels.append(1 if label == name else 0)
    return np.array(dists), np.array(labels)


def find_eer(dists, labels):
    fpr, tpr, thr = roc_curve(labels, -dists)  # مسافة أقل = تشابه أعلى
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return (fpr[i] + fnr[i]) / 2, -thr[i], fpr, tpr


def report(title, y_true, y_pred, names):
    print(f"\n===== {title} =====")
    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(
        y_true, y_pred, labels=names + ["Unknown"], zero_division=0))


def build_splits(data, unknown_ratio, rng):
    """يرجع (enrolled, unknown_people, db, val_probes, test_probes)"""
    people = sorted(data)
    n_unknown = max(1, round(len(people) * unknown_ratio))
    unknown_people = rng.sample(people, n_unknown)
    enrolled = [p for p in people if p not in unknown_people]

    db, val_probes, test_probes = {}, [], []
    for p in enrolled:
        tr, va, te = split_person(data[p], rng)
        db[p] = make_template(tr)
        val_probes += [(p, e) for e in va]
        test_probes += [(p, e) for e in te]
    # أشخاص غير مسجّلين: نصف صورهم للتحقق ونصفها للاختبار
    for p in unknown_people:
        embs = data[p][:]
        rng.shuffle(embs)
        half = len(embs) // 2
        val_probes += [("Unknown", e) for e in embs[:half]]
        test_probes += [("Unknown", e) for e in embs[half:]]
    return enrolled, unknown_people, db, val_probes, test_probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval_data")
    ap.add_argument("--threshold", type=float, default=0.5, help="الحالي في التطبيق")
    ap.add_argument("--unknown-ratio", type=float, default=0.2,
                    help="نسبة الأشخاص غير المسجّلين (لاختبار Unknown)")
    ap.add_argument("--no-detect", action="store_true", help="الصور مقصوصة مسبقًا")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # استيراد متأخر حتى لا يُحمَّل TensorFlow إلا عند التشغيل الفعلي
    from detector import FaceDetector
    from embedder import FaceEmbedder

    rng = random.Random(args.seed)
    detector, embedder = FaceDetector(), FaceEmbedder()

    print("Extracting embeddings:")
    data = load_embeddings(args.data, detector, embedder, detect=not args.no_detect)
    data = {k: v for k, v in data.items() if len(v) >= 5}
    if len(data) < 3:
        raise SystemExit("Need at least 3 people with 5+ usable images each.")

    enrolled, unknown_people, db, val_probes, test_probes = build_splits(
        data, args.unknown_ratio, rng)
    print(f"\nEnrolled: {len(enrolled)} people | Unknown: {len(unknown_people)} people")
    print(f"Validation probes: {len(val_probes)} | Test probes: {len(test_probes)}")

    # ضبط الـ threshold على مجموعة التحقق فقط
    d_val, l_val = verification_scores(val_probes, db)
    eer, eer_thr, _, _ = find_eer(d_val, l_val)
    print(f"\n[Validation] EER = {eer:.4f} at threshold = {eer_thr:.3f}")

    # النتائج النهائية على مجموعة الاختبار
    for title, thr in [(f"Test @ current threshold ({args.threshold})", args.threshold),
                       (f"Test @ tuned threshold ({eer_thr:.3f})", eer_thr)]:
        y_true, y_pred = run_identification(test_probes, db, thr)
        report(title, y_true, y_pred, enrolled)

    # ROC على مجموعة الاختبار
    d_te, l_te = verification_scores(test_probes, db)
    eer_te, _, fpr, tpr = find_eer(d_te, l_te)
    print(f"[Test] EER = {eer_te:.4f} | AUC = {auc(fpr, tpr):.4f}")

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (EER = {eer_te:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc.png", dpi=150)
    print("ROC curve saved to roc.png")


if __name__ == "__main__":
    main()
