from deepface import DeepFace


class Race_Model:
    def __init__(self):
        pass

    def predict_race(self, face_image):
        try:
            result = DeepFace.analyze(
                face_image,
                actions=["race"],
                enforce_detection=False,
                silent=True
            )
            if isinstance(result, list):
                return result[0]["dominant_race"]
            return result["dominant_race"]
        except Exception as e:
            print(f"Race error: {e}")
            return "N/A"
