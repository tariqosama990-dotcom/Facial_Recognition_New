import cv2
import numpy as np
from keras_facenet import FaceNet


class FaceEmbedder:
    def __init__(self, input_size=(160, 160)):
        self.model = FaceNet()
        self.input_size = input_size

    @staticmethod
    def crop(frame_bgr, box):
        x, y, w, h = box
        return frame_bgr[y:y + h, x:x + w]

    def preprocess(self, face_bgr):
        """BGR -> RGB ثم تغيير الحجم إلى 160x160"""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, self.input_size)

    def embed(self, face_bgr):
        """ترجع متجه البصمة (512,) لوجه مقصوص"""
        img = self.preprocess(face_bgr)
        return self.model.embeddings(np.expand_dims(img, axis=0))[0]

    def embed_from_frame(self, frame_bgr, box):
        return self.embed(self.crop(frame_bgr, box))