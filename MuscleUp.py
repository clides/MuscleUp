import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video Feed
cap = cv2.VideoCapture("MuscleUp.mp4")

prev_frame_time = 0
new_frame_time = 0

# Setup MediaPipe Instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        success, frame = cap.read()

        # Recolor Image (mediapipe expects RGB but cv2 gives BGR)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writable = False # Saves memory

        # Make Detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw Landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        

        cv2.imshow("Image", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
