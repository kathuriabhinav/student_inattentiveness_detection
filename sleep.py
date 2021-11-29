from scipy.spatial import distance as dist
import cv2
import numpy as np
import pyttsx3

engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate',rate - 50)

#Dlib eye landmarks:        
    #L38,42 #R44,48     
    #L39,41 #R45,47     
    #L37,40 #R43,46     

# Eye mediapipe landmarks:
#Left:                           #Right:
#          159                            386
#     160        158                 385       387
# 33                  133       362                 263       
#     144        153                 380       373
#          145                            374

EYE_AR_THRESH = 0.10  #threshold for blink
EYE_AR_CONSEC_FRAMES = 15 #consecutive considered true
MOUTH_THRESHOLD = 0.9

def eye_aspect_ratio(landmarks, dir):
    ear = 0
    try:
        if landmarks.shape[0] >=314:
            if dir == "left":
                A = dist.euclidean(landmarks[160,:], landmarks[144,:])
                B = dist.euclidean(landmarks[159,:], landmarks[145,:])
                C = dist.euclidean(landmarks[158,:], landmarks[153,:])
                D = dist.euclidean(landmarks[33,:], landmarks[133,:]) 
            elif dir == "right":
                A = dist.euclidean(landmarks[385,:], landmarks[380,:])
                B = dist.euclidean(landmarks[386,:], landmarks[374,:])
                C = dist.euclidean(landmarks[387,:], landmarks[373,:])
                D = dist.euclidean(landmarks[362,:], landmarks[263,:]) 
            ear = (A+B+C)/(3*D)
    except:
        pass
    return ear

# Mouth mediapipe landmarks: 
outer_bottom = [61,146,91,181,84,17,314,405,321,375,291]
outer_top = [61,185,40,39,37,0,267,269,270,409,291]
inner_bottom = [78,95,88,178,87,14,317,402,318,324,308]
inner_top = [78,191,80,81,82,13,312,311,310,415,308]

def mouth_aspect_ratio(landmarks):
    mar = 0
    try:
        if landmarks.shape[0] >=314: 
                A = dist.euclidean(landmarks[37,:], landmarks[84,:])   # 51, 59 # media(37,84)
                B = dist.euclidean(landmarks[0,:], landmarks[17,:])    # 52, 58 #media(0,17)
                C = dist.euclidean(landmarks[267,:], landmarks[314,:]) # 53, 57 # media(267,314)        
                D = dist.euclidean(landmarks[61,:], landmarks[291,:])  # 49, 55 #media(61,291)
                mar = (A+B+C)/(3*D)
    except:
        pass
    return mar

def sleep_main(faces,frame,sleep_flag,yawn_flag,count_mouth,counter,total,total_yawn):
    if not faces:
        return sleep_flag,yawn_flag,count_mouth,counter,total,total_yawn
    
    face = None

    for i in range(len(faces)):
        if (faces[i].name == "verified" and type(faces[i].hland) is np.ndarray):
            face = faces[i]
            break

    if not face:
        for i in range(len(faces)):
            if type(faces[i].hland) is np.ndarray:
                face = faces[i]
                break

    if face:
        landmarks = np.array(face.landmarks)[:, :2]
        leftEAR = eye_aspect_ratio(landmarks, "left")
        rightEAR = eye_aspect_ratio(landmarks, "right")
        mouthEAR = mouth_aspect_ratio(landmarks)
        ear = (leftEAR + rightEAR) / 2.0

        if mouthEAR > MOUTH_THRESHOLD:
            count_mouth += 1
            if count_mouth >= 10:
                if yawn_flag < 0:
                    face.framesleepy=1
                    print("You are yawning")
                    yawn_flag = 1
                    total_yawn += 1
                    cv2.putText(frame, "Yawn Detected", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 190), 2)
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
                    face.framesleepy=1
                    print("You are sleeping.")
                    cv2.putText(frame, "Sleep Detected", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 190), 2)
                    sleep_flag = 1
                    total += 1
            else:
                sleep_flag = -1
        else:
            counter = 0
            sleep_flag = -1

        cv2.putText(frame, "Total Sleeps: {}".format(total), (15,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 190, 190), 2)
        cv2.putText(frame, "Total Yawns: {}".format(total_yawn), (15,230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 190, 190), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (15,260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 190, 190), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mouthEAR), (15,290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 190, 190), 2)
        if total > 2:
            face.sleepy=1
            face.framesleepy = 1
            print("playing sound")
            engine.say("You are drowsy. Please be Attentive")
            engine.runAndWait()
            total = 0
            total_yawn = 0
        elif total_yawn > 2:
            face.sleepy=1
            face.framesleepy = 1
            print("playing sound")
            engine.say("You are drowsy. Please be Attentive")
            engine.runAndWait()
            total = 0
            total_yawn = 0
        elif total+total_yawn > 2:
            face.sleepy=1
            face.framesleepy = 1
            print("playing sound")
            engine.say("You are drowsy. Please be Attentive")
            engine.runAndWait()
            total = 0
            total_yawn = 0

    return sleep_flag,yawn_flag,count_mouth,counter,total,total_yawn