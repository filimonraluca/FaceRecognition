import cv2
import dlib
import numpy
import math
from scipy.spatial import distance

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 48
WIDTH = 320
HEIGHT = 240

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

blinks = 0


def euclidean_dist(x,y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

def eye_aspect_ratio(eye):
    global blinks
    a = euclidean_dist(eye[1], eye[5])
    b =euclidean_dist(eye[2], eye[4])
    c = euclidean_dist(eye[0], eye[3])
    return (a + b) / (2.0 * c)


cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cam.read()
    if ret is False:
        break
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = []
        right_eye = []
        for x in range(0, 6):
            left_eye.append((landmarks.part(36 + x).x, landmarks.part(36 + x).y))
            right_eye.append((landmarks.part(42 + x).x, landmarks.part(42 + x).y))
        if (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2 < EYE_AR_THRESH:
            blinks += 1
            print("BLINK " + str(blinks))
    cv2.imshow("Frame", gray)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
        break
cam.release()
cv2.destroyAllWindows()
