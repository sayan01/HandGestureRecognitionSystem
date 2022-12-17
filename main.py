import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipe
mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# initialize tensorflow
model = load_model('mp_hand_gesture')
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# initialize webcam using opencv
capture = cv2.VideoCapture(0)

while 1:
    # read frame from webcam indefinitely
    success, frame = capture.read()
    frame = cv2.flip(frame, 1)
    # show frame on screen
    cv2.imshow("Press q to exit", frame)
    if cv2.waitKey(1) == ord('q'):
        break # if user pressed q then quit

capture.release()
cv2.destroyAllWindows()
