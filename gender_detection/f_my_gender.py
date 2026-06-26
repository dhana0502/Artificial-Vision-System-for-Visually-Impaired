from deepface import DeepFace


class Gender_Model:
    def __init__(self):
        pass

    def predict_gender(self, face_image):
        try:
            result = DeepFace.analyze(
                face_image,
                actions=["gender"],
                enforce_detection=False,
                silent=True
            )
            if isinstance(result, list):
                return result[0]["dominant_gender"]
            return result["dominant_gender"]
        except Exception as e:
            print(f"Gender error: {e}")
            return "N/A"
