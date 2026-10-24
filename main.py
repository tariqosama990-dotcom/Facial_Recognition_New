from detector import FaceDetector
from embedder import FaceEmbedder
from matcher import FaceMatcher
from database import FaceDatabase
from gui import FaceApp

print("Loading the FaceNet model...")
embedder = FaceEmbedder()
print("The model has been successfully loaded.")

app = FaceApp(
    detector=FaceDetector(),
    embedder=embedder,
    matcher=FaceMatcher(threshold=0.5),
    database=FaceDatabase(),
)
app.run()