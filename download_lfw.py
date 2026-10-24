import os

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

OUT_DIR = "eval_lfw"

# slice_=None و resize=1.0 يعطيان الصورة الكاملة (250x250) بدل المقصوصة الضيقة
lfw = fetch_lfw_people(min_faces_per_person=15, color=True,
                       slice_=None, resize=1.0)
print("Images:", len(lfw.images), "| People:", len(lfw.target_names))

for img, label in zip(lfw.images, lfw.target):
    name = lfw.target_names[label].replace(" ", "_")
    folder = os.path.join(OUT_DIR, name)
    os.makedirs(folder, exist_ok=True)
    n = len(os.listdir(folder)) + 1

    # sklearn يرجّع قيمًا بين 0 و1 (أو 0 و255 حسب الإصدار)
    if img.max() <= 1.0:
        img = img * 255.0
    img_u8 = np.clip(img, 0, 255).astype("uint8")

    cv2.imwrite(os.path.join(folder, f"{n}.jpg"),
                cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))

print(f"Saved to {OUT_DIR}/")