import cv2
import imutils
import time
import numpy as np
import f_Face_info
import audio_output


def run_on_webcam():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open webcam")
        return

    print("Press q to quit")
    while True:
        start_time = time.time()
        ret, frame = cam.read()
        if not ret:
            break

        # Resize
        frame = imutils.resize(frame, width=720)

        # Force correct format
        frame = np.array(frame, dtype=np.uint8)
        frame = np.ascontiguousarray(frame)

        out = f_Face_info.get_face_info(frame)
        res_img = f_Face_info.bounding_box(out, frame)

        # Speak out info for the first detected face
        if out and len(out[0].get("bbx_frontal_face", [])) > 0:
            audio_output.describe_face(out[0])

        elapsed = time.time() - start_time
        fps = 1 / elapsed if elapsed > 0 else 0
        cv2.putText(res_img, f"FPS:{round(fps,2)}",
                    (10, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 2)

        cv2.imshow("Face Info", res_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_on_webcam()
