from deepface import DeepFace


class predict_emotions:
    def __init__(self):
        pass

    def get_emotion(self, img, boxes_face):
        emotions = []
        if len(boxes_face) == 0:
            return boxes_face, emotions

        for box in boxes_face:
            try:
                x0, y0, x1, y1 = box
                face_crop = img[y0:y1, x0:x1]
                if face_crop.size == 0:
                    emotions.append("N/A")
                    continue

                result = DeepFace.analyze(
                    face_crop,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True
                )
                if isinstance(result, list):
                    emotions.append(result[0]["dominant_emotion"])
                else:
                    emotions.append(result["dominant_emotion"])
            except Exception as e:
                print(f"Emotion error: {e}")
                emotions.append("N/A")

        return boxes_face, emotions
