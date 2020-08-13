from train_nn import *
import cv2
import numpy as np
import dlib
from math import hypot
import time
from keras.models import load_model
from gaze_tracking import GazeTracking

#---CONSTS---
lastFrameTime = time.time()
gaze = GazeTracking()

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
predictor = load_model('my_model.h5')

#---Image feed---
while True:
    _, frame = cap.read()
    gaze.refresh(frame) #gazetracking analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#---Texts---
    #fps text
    cv2.putText(frame, "FPS: %.2f" %(1/(time.time() - lastFrameTime)), (530, 20), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
    lastFrameTime = time.time()
    
    #gaze text
    frame = gaze.annotated_frame()
    text_oy = ""
    text_ox = ""

    if gaze.is_up():
        text_ox = "up"
    elif gaze.is_down():
        text_ox = "down"
    elif gaze.is_ox_center():
        text_ox = "center"
    elif (not gaze.is_right() and not gaze.is_left() and not gaze.is_oy_center() and not gaze.is_ox_center() and not gaze.is_up() and not gaze.is_down()):
        text_oy = "blink"
        text_ox = "blink"
    #if gaze.is_blinking():
        #text_oy = "dwn/blk"
    if gaze.is_right():
        text_oy = "right"
    elif gaze.is_left():
        text_oy = "left"
    elif gaze.is_oy_center():
        text_oy = "center"

    cv2.putText(frame, "OX: " + text_oy, (30, 20), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
    cv2.putText(frame, "OY: " + text_ox, (120, 20), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
    
    #---Process---
    faces = detector.detectMultiScale(gray, 1.25, 6)
    
    for (x, y, w, h) in faces:
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]    
        
        #---Gaze
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (30, 50), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (30, 35), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
        #---EndofGaze

        # Normalize to match the input format of the model - Range of pixel to [0, 1]
        gray_normalized = gray_face / 255

        # Resize it to 96x96 to match the input format of the model
        original_shape = gray_face.shape # A Copy for future usage
        face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized = face_resized.reshape(1, 96, 96, 1)

        # Predicting the keypoints using the model
        keypoints = predictor.predict(face_resized)

        # De-Normalize the keypoints values
        keypoints = keypoints * 48 + 48

        # Map the Keypoints back to the original image
        face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)

        # Pair them together
        points = []
        for i, co in enumerate(keypoints[0][0::2]):
            points.append((co, keypoints[0][1::2][i]))

        for keypoint in points:
            cv2.circle(face_resized_color, keypoint, 1, (0,255,0), 1)

        frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)
        

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
        
# Cleanup the cap and close any open windows
cap.release()
cv2.destroyAllWindows()