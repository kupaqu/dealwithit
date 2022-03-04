import cv2
import numpy as np

def detectAndDisplay(frame, face_classifier, eye_classifier, glasses, scaleFactor=None, minNeighbors=None):

    result = frame.copy()

    # ищем лица
    faces_rects = face_classifier.detectMultiScale(
        result, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in faces_rects:
        face = result[y:y+h, x:x+w]
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 255))

        # ищем глаза внутри лиц
        eyes_rects = eye_classifier.detectMultiScale(
            face, scaleFactor=scaleFactor,
            minNeighbors=minNeighbors, minSize=(5, 5))

        # перспектива
        if len(eyes_rects) == 2:
            eyes_rects = sorted(eyes_rects, key=lambda x: x[0]) # сортировка по горизонтали
            # print(eyes_rects)
            upper_left = np.array([eyes_rects[0][0], eyes_rects[0][1]], dtype=np.float32)
            bottom_left = np.array([eyes_rects[0][0], eyes_rects[0][1] + eyes_rects[0][3]], dtype=np.float32)
            bottom_right = np.array([eyes_rects[1][0] + eyes_rects[1][2], eyes_rects[1][1] + eyes_rects[1][3]], dtype=np.float32)
            upper_right = np.array([eyes_rects[1][0] + eyes_rects[1][2], eyes_rects[1][1]], dtype=np.float32)
            output_pts = np.float32([upper_left, bottom_left, bottom_right, upper_right])

            input_pts = np.float32([
                [0, 0],
                [0, glasses.shape[0]],
                [glasses.shape[1], glasses.shape[0]],
                [glasses.shape[1], 0]
            ])

            M = cv2.getPerspectiveTransform(input_pts, output_pts)

            glasses = cv2.warpPerspective(glasses, M, (face.shape[1], face.shape[0]), cv2.INTER_LINEAR)
            gw, gh, gc = glasses.shape
            # print(glasses.shape)
            for i in range(0, gw):
                for j in range(0, gh):
                    face[i, j] = glasses[i, j]

        for (x2, y2, w2, h2) in eyes_rects:
            cv2.rectangle(face, (x2, y2), (x2+w2, y2+h2), (255, 255, 255))

    return result

glasses = cv2.imread("dealwithit.png")

face_cascade = "haarcascade_frontalface_default.xml"
eye_cascade = "haarcascade_eye.xml"

face_classifier = cv2.CascadeClassifier(face_cascade)
eye_classifier = cv2.CascadeClassifier(eye_cascade)

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

while cam.isOpened():
    _, image = cam.read()
    image = cv2.flip(image, 1)

    result = detectAndDisplay(image, face_classifier, eye_classifier, glasses, 1.2, 7)

    cv2.imshow("Camera", result)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()