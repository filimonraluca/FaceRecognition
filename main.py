import cv2
import dlib
import math
import numpy as np
import frame_constants
from eye_detection import EyeDetection


class FaceRecognition:
    def __init__(self):
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.landmarks = None
        self.frame = None
        self.drawing_frame = None
        self.left_eye = None
        self.right_eye = None

    def draw_face_points(self):
        for i in range(0, 68):
            x_p, y_p = self.landmarks.part(i).x, self.landmarks.part(i).y
            cv2.circle(self.drawing_frame, (x_p, y_p), 2, (255, 0, 0), -1)

    def face_recognition(self):
        while True:
            ret, self.frame = self.cam.read()
            if ret is False:
                break
            self.frame = cv2.resize(self.frame, (frame_constants.WIDTH, frame_constants.HEIGHT))
            self.drawing_frame = self.frame.copy()
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            faces = self.detector(gray)
            for face in faces:
                self.landmarks = self.predictor(gray, face)
                self.draw_face_points()

                self.left_eye = [(self.landmarks.part(36 + x).x, self.landmarks.part(36 + x).y) for x in range(0, 6)]
                self.right_eye = [(self.landmarks.part(42 + x).x, self.landmarks.part(42 + x).y) for x in range(0, 6)]
                eye_dec = EyeDetection(self.frame, self.drawing_frame, self.left_eye, self.right_eye).detect_eye()
            cv2.imshow("Frame", self.drawing_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
                break
        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr_obj = FaceRecognition()
    fr_obj.face_recognition()
