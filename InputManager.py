import cv2
import numpy as np


def TakeVideo(source):
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if ret is False:
            return ret
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return ret


def TakeImage(source):
    if source == 0:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                return ret
            img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        cap.release()
    else:

        img = cv2.imread(source, cv2.IMREAD_COLOR)
        if img is None:
            return False
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return True
