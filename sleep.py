from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pyttsx3

engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate',rate - 50)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A+B)/(2*C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2],mouth[10])
    B = dist.euclidean(mouth[3],mouth[9])
    C = dist.euclidean(mouth[4],mouth[8])
    mar = (A+B+C)/3
    return mar


EYE_AR_THRESH = 0.23  #threshold for blink
EYE_AR_CONSEC_FRAMES = 25 #consecutive considered true


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def sleep_main(faces,frame,sleep_flag,yawn_flag,count_mouth,counter,total,total_yawn):
    # frame = vs.read()
    
    if not faces:
        cv2.putText(frame, "return 1", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return sleep_flag,yawn_flag,count_mouth,counter,total,total_yawn
    if len(faces)!=1:
        cv2.putText(frame, "return 2", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return sleep_flag,yawn_flag,count_mouth,counter,total,total_yawn

    if len(faces)==1:    
    
        temp = imutils.resize(frame, width = 640)
        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

        rects = detector(gray,0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            mouth = shape[mStart: mEnd]
            mouthEAR = mouth_aspect_ratio(mouth)

            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)

            if mouthEAR > 30:
                count_mouth += 1
                if count_mouth >= 10:
                    if yawn_flag < 0:
                        faces[0].framesleepy=1
                        print("You are yawning")
                        yawn_flag = 1
                        total_yawn += 1
                        cv2.putText(frame, "Yawn Detected", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    else:
                        yawn_flag = 1
                else:
                    yawn_flag = -1
            else:
                count_mouth = 0
                yawn_flag = -1
            if ear <  EYE_AR_THRESH:
                counter += 1

                if counter >= EYE_AR_CONSEC_FRAMES:
                    if sleep_flag < 0:
                        faces[0].framesleepy=1
                        print("You are sleeping.")
                        cv2.putText(frame, "Sleep Detected", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        sleep_flag = 1
                        total += 1
                else:
                    sleep_flag = -1
            else:
                counter = 0
                sleep_flag = -1

            cv2.putText(frame, "Total Sleeps: {}".format(total), (15,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, "Total Yawns: {}".format(total_yawn), (15,230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (15,260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, "MAR: {:.2f}".format(mouthEAR), (15,290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if total > 2:
                faces[0].sleepy=1
                print("playing sound")
                engine.say("You are drowsy. Please be Attentive")
                engine.runAndWait()
                total = 0
                total_yawn = 0
            elif total_yawn > 2:
                faces[0].sleepy=1
                print("playing sound")
                engine.say("You are drowsy. Please be Attentive")
                engine.runAndWait()
                total = 0
                total_yawn = 0
            elif total+total_yawn > 2:
                faces[0].sleepy=1
                print("playing sound")
                engine.say("You are drowsy. Please be Attentive")
                engine.runAndWait()
                total = 0
                total_yawn = 0

    return sleep_flag,yawn_flag,count_mouth,counter,total,total_yawn