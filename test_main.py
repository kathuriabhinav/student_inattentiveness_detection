import cv2
import utils
import numpy as np
from face_detector import detect_faces
from face_recognition import Recognizer
from face_spoofing import spoof_detector
from face_landmarks import detect_landmarks
from head_pose import headpose_est
from mouth_detector import mouth_open
from face_detector import detect_faces
from sleep import sleep_main

sleep_flag = 0
yawn_flag = 0
count_mouth = 0

counter = 0
total = 0
total_yawn = 0

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()
    if not ret :
        break
    
    faces =  detect_faces(frame, confidence = 0.7)
    if faces:
        if len(faces)==1:
            hland = detect_landmarks(frame, faces) 
            frame,sleep_flag,yawn_flag,count_mouth,counter,total,total_yawn = sleep_main(frame,sleep_flag,yawn_flag,count_mouth,counter,total,total_yawn)
        frame = utils.print_faces(frame, faces)
    cv2.imshow('Monitoring On',  frame)
            
    if cv2.waitKey(1) & 0xFF == 27: 
        break
cap.release()