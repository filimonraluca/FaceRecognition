import cv2
import math
import numpy
import frame_constants

prev_ratio = 0


class EyeDetection:
    def __init__(self, frame, drawing_frame, left_eye, right_eye):
        self.frame = frame
        self.drawing_frame = drawing_frame
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.ratio = None

    @staticmethod
    def euclidian_dist(x, y):
        return math.hypot((x[0] - y[0]), (x[1] - y[1]))

    @staticmethod
    def midpoint(x, y):
        return int((x[0] + y[0]) / 2), int((x[1] + y[1]) / 2)

    def draw_eye_line(self, eye):
        cv2.line(self.drawing_frame, eye[0], eye[3], (0, 255, 0), 1)
        center_top = self.midpoint(eye[1], eye[2])
        center_buttom = self.midpoint(eye[4], eye[5])
        cv2.line(self.drawing_frame, center_top, center_buttom, (0, 255, 0), 1)

    def eye_aspect_ratio(self, eye):
        a = self.euclidian_dist(eye[1], eye[5])
        b = self.euclidian_dist(eye[2], eye[4])
        c = self.euclidian_dist(eye[0], eye[3])
        return (a + b) / (2.0 * c)

    def detect_eye(self):
        self.draw_eye_line(self.left_eye)
        self.blinking()
        self.gaze_detection()

    def blinking(self):
        global prev_ratio
        self.ratio = (self.eye_aspect_ratio(self.left_eye) + self.eye_aspect_ratio(self.right_eye)) / 2.0
        cv2.putText(self.drawing_frame, str(self.ratio), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        if prev_ratio:
            if prev_ratio - self.ratio > 0.03:
                cv2.putText(self.drawing_frame, "blink", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
        prev_ratio = self.ratio

    def get_eye_region(self, eye):
        eye_region = numpy.array(eye, numpy.int32)
        min_x = numpy.min(eye_region[:, 0])
        max_x = numpy.max(eye_region[:, 0])
        min_y = numpy.min(eye_region[:, 1])
        max_y = numpy.max(eye_region[:, 1])
        return eye_region, (min_x, max_x, min_y, max_y)

    @staticmethod
    def select_eye(eye_rectangle, masked_frame):
        eye_frame = masked_frame[eye_rectangle[2]:eye_rectangle[3],
                    eye_rectangle[0]:eye_rectangle[1]]
        eye_frame = cv2.resize(eye_frame, None, fx=10, fy=10)
        _, threshhold_eye = cv2.threshold(eye_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshhold_eye

    def gaze_detection(self):
        left_eye_region, left_eye_rectangle = self.get_eye_region(self.left_eye)
        right_eye_region, right_eye_rectangle = self.get_eye_region(self.right_eye)
        mask = numpy.zeros((frame_constants.HEIGHT, frame_constants.WIDTH), numpy.uint8)
        cv2.polylines(mask, [left_eye_region], True, (0, 0, 255), 1)
        cv2.fillPoly(mask, [left_eye_region], 255)
        cv2.polylines(mask, [right_eye_region], True, (0, 0, 255), 1)
        cv2.fillPoly(mask, [right_eye_region], 255)
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        masked_frame = cv2.bitwise_and(gray, gray, mask=mask)
        cv2.imshow("Eye masked", masked_frame)
        left_eye_frame = self.select_eye(left_eye_rectangle, masked_frame)
        right_eye_frame = self.select_eye(right_eye_rectangle, masked_frame)

        cv2.imshow("left eye", left_eye_frame)
        cv2.imshow("right eye", right_eye_frame)


if __name__ == '__main__':
    pass
