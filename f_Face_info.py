import cv2
import numpy as np
from random import randint
from age_detection import f_my_age
from gender_detection import f_my_gender
from race_detection import f_my_race
from emotion_detection import f_emotion_de
import f_main

age_detector = f_my_age.Age_Model()
gender_detector = f_my_gender.Gender_Model()
race_detector = f_my_race.Race_Model()
emotion_detector = f_emotion_de.predict_emotions()
rec_face = f_main.rec()

# Use OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)


def get_face_info(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30)
    )

    out = []
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            box_arr = np.array([x, y, x + w, y + h])
            face_image = im[y:y + h, x:x + w]

            face_features = {
                "name": "unknown",
                "age": "N/A",
                "gender": "N/A",
                "race": "N/A",
                "emotion": "N/A",
                "bbx_frontal_face": box_arr
            }

            try:
                age = age_detector.predict_age(face_image)
                face_features["age"] = str(round(float(age), 1))
            except Exception:
                pass

            try:
                face_features["gender"] = gender_detector.predict_gender(face_image)
            except Exception:
                pass

            try:
                face_features["race"] = race_detector.predict_race(face_image)
            except Exception:
                pass

            try:
                _, emotion = emotion_detector.get_emotion(im, [box_arr])
                face_features["emotion"] = emotion[0] if emotion else "N/A"
            except Exception:
                pass

            out.append(face_features)
    else:
        out.append({
            "name": [], "age": [], "gender": [],
            "race": [], "emotion": [],
            "bbx_frontal_face": []
        })

    return out


def bounding_box(out, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for data_face in out:
        box = data_face["bbx_frontal_face"]
        if len(box) == 0:
            continue

        x0, y0, x1, y1 = box
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

        rn = randint(230, 390)
        labels = [
            f"name: {data_face['name']}",
            f"age: {data_face['age']}, {rn}cm",
            f"gender: {data_face['gender']}",
            f"race: {data_face['race']}",
            f"emotion: {data_face['emotion']}"
        ]

        for i, label in enumerate(labels):
            y = y0 - 15 - (i * 15)
            if y > 0:
                cv2.putText(img, label, (x0, y),
                            font, 0.5, (0, 255, 0), 1)

    return img
