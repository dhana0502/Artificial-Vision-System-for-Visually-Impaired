from deepface import DeepFace


class Age_Model:
    def __init__(self):
        pass

    def predict_age(self, face_image):
        try:
            result = DeepFace.analyze(
                face_image,
                actions=["age"],
                enforce_detection=False,
                silent=True
            )
            if isinstance(result, list):
                return result[0]["age"]
            return result["age"]
        except Exception as e:
            print(f"Age error: {e}")
            return 0
