from scipy.spatial.distance import cosine


class FaceMatcher:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def recognize(self, live_vector, db):
        """db: dict اسم -> بصمة. ترجع (identity, min_dist)"""
        if not db:
            return "Unknown", 1.0

        min_dist = float("inf")
        identity = "Unknown"

        for name, saved_vector in db.items():
            dist = cosine(live_vector, saved_vector)
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > self.threshold:
            identity = "Unknown"

        return identity, min_dist