import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipe
mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
mpStyles = mp.solutions.drawing_styles

# initialize tensorflow
model = load_model('mp_hand_gesture')
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# initialize webcam using opencv
capture = cv2.VideoCapture(0)

while capture.isOpened():
    # read frame from webcam indefinitely
    success, frame = capture.read()
    if not success: break # break if fucked
    frame = cv2.flip(frame, 1)
    x , y, c = frame.shape
    # show frame on screen
    if cv2.waitKey(1) == ord('q'):
        break # if user pressed q then quit

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    
    if result.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append([int(landmark.x * x), int(landmark.y * y)])
            mpDraw.draw_landmarks(frame, hand_landmarks, mphands.HAND_CONNECTIONS, mpStyles.get_default_hand_landmarks_style(), mpStyles.get_default_hand_connections_style())
        prediction = model.predict([landmarks])
        classid = np.argmax(prediction)
        className = classNames[classid] if classid in range(len(classNames)) else ""
        cv2.putText(frame, f'{classid} {className}',(10,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow("Press q to exit", frame)

capture.release()
cv2.destroyAllWindows()
