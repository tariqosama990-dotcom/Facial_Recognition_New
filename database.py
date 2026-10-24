import os
import pickle

import cv2


class FaceDatabase:
    def __init__(self, db_path="face_embeddings.pkl", dataset_dir="dataset", embedding_size=512):
        self.db_path = db_path
        self.dataset_dir = dataset_dir
        self.embedding_size = embedding_size
        self.known_embeddings = {}

        os.makedirs(self.dataset_dir, exist_ok=True)
        self.image_count = len(os.listdir(self.dataset_dir))
        self.load()

    def load(self):
        """تحميل البصمات مع تجاهل غير المتوافقة (نفس سلوك aa.py)"""
        self.known_embeddings = {}
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                loaded_db = pickle.load(f)

            for name, embedding in loaded_db.items():
                if len(embedding) == self.embedding_size:
                    self.known_embeddings[name] = embedding
                else:
                    print(f" Ignored [{name}] because embedding size ({len(embedding)}) "
                          f"is incompatible with current model ({self.embedding_size}).")

            print(f"A person has been loaded {len(self.known_embeddings)} from memory.")
        else:
            print("The database is currently empty.")

    def save_person(self, name, face_rgb, embedding):
        """حفظ صورة الوجه (RGB 160x160) والبصمة. نفس منطق save_new_person"""
        self.image_count += 1
        file_path = f"{self.dataset_dir}/{name}_{self.image_count}.jpg"
        bgr_face = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, bgr_face)

        self.known_embeddings[name] = embedding
        with open(self.db_path, "wb") as f:
            pickle.dump(self.known_embeddings, f)