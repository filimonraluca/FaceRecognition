import cv2
import dlib
import math

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
WIDTH = 320
HEIGHT = 240

def euclidian_dist(x, y):
    return math.hypot((x[0] - y[0]), (x[1] - y[1]))


def midpoint(x, y):
    return int((x[0] + y[0]) / 2), int((x[1] + y[1]) / 2)


def eye_aspect_ratio(eye):
    a = euclidian_dist(eye[1], eye[5])
    b = euclidian_dist(eye[2], eye[4])
    c = euclidian_dist(eye[0], eye[3])
    return (a + b) / (2.0 * c)


def draw_eye_line(eye):
    cv2.line(frame, eye[0], eye[3], (0, 255, 0), 1)
    center_top = midpoint(eye[1], eye[2])
    center_buttom = midpoint(eye[4], eye[5])
    cv2.line(frame, center_top, center_buttom, (0, 255, 0), 1)

prev_ratio = None

while True:
    ret, frame = cam.read()
    if ret is False:
        break
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        for i in range(0, 68):
            x_p, y_p = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x_p, y_p), 2, (255, 0, 0), -1)
        left_eye = []
        right_eye = []
        for x in range(0, 6):
            left_eye.append((landmarks.part(36 + x).x, landmarks.part(36 + x).y))
            right_eye.append((landmarks.part(42 + x).x, landmarks.part(42 + x).y))
        draw_eye_line(left_eye)
        draw_eye_line(right_eye)
        ratio = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        cv2.putText(frame, str(ratio), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
        if prev_ratio:
            if prev_ratio-ratio > 0.05:
                cv2.putText(frame, "blink", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
        prev_ratio = ratio

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
        break
cam.release()
cv2.destroyAllWindows()
