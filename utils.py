import cv2
import time
import numpy as np
import mediapipe as mp
from face_detector import detect_faces
font = cv2.FONT_HERSHEY_SIMPLEX 

def print_fps(frame, pTime):
    cTime = time.time()
    fps = 1/(cTime - pTime[0])
    pTime[0] = cTime
    cv2.putText(frame, f"FPS : {int(fps)}", (15,30),font, 0.5, (255,0,0),2)
    return frame

def print_faces(frame, faces):
    for face in faces:
        x,y,w,h = face.bbox
        bool_flag=0;
        #Face detection
        cv2.rectangle(frame, face.bbox, (0, 0, 153), 2)
        cv2.putText(frame, "c:"+str(round(face.confidence[0],4)),(x+w+5, y+28), cv2.FONT_HERSHEY_PLAIN, 1, (153,0,0), 1)
        
        #Face Recognition
        if face.name:
            cv2.putText(frame, face.name, (x, y-5),cv2.FONT_HERSHEY_PLAIN, 1, (204,0,0),2)
            cv2.putText(frame, "d:"+str(round(face.distance,4)),(x+w+5, y+46), cv2.FONT_HERSHEY_PLAIN, 1, (153,0,0), 1)
        
        #Face Spoofing    
        if face.spoof!=None:
            cv2.putText(frame,"real",(x+w+5, y+10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2) if face.spoof else cv2.putText(frame,"spoof",(x+w+5, y+10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            cv2.putText(frame, "spf:"+str(round(face.spoof_score,4)),(x+w+5, y+64), cv2.FONT_HERSHEY_PLAIN, 1, (153,0,0), 1)
        
        #Face landmarks
        if face.landmarks:
            for (x1, y1,_) in face.landmarks[:468]:
                cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)
            for (x1, y1,_) in face.landmarks[468:]:
                cv2.circle(frame, (x1, y1), 1, (0, 128, 255), -1)
        
        #Mouth tracker
        if face.mouth:
            cv2.putText(frame, "MAR:"+str(round(face.mouth.mar,4)),(x+w+5, y+82), cv2.FONT_HERSHEY_PLAIN, 0.9, (153,0,0), 1)
            cv2.putText(frame, face.mouth.status, (15,55), font, 0.5, (255,0,0),2)
        
    return frame


def register_user(fr, num_pics = 5, skipr = False):  # Here model is Face_recogntion model
    user_name = "User"
    cam = cv2.VideoCapture("inhouse_dataset/abhi.mp4")
    cv2.namedWindow('Face registration')
    count = 0
    input_embeddings = {}
    
    while count<num_pics:  
        ret, frame = cam.read()
        if ret:

            cv2.putText(frame, 'Press r to capture image, {}/{} captures done'.format(count,num_pics),(30,60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
            cv2.imshow('Face registration', frame)
        
            if skipr or cv2.waitKey(1) & 0xFF == ord('r'):
                # capturing image 
                faces =  detect_faces(frame, confidence = 0.7)
                if len(faces)!=1:
                    print('No face detected or Multiple faces detected. Please try again.')
                else:
                    face_img = faces[0].img
                    input_embeddings[user_name] = fr.face_to_embedding(face_img)
                    count+=1
                    # saving image as use_image.jpg for further face verification
                    cv2.imwrite("captures/{}_{}.jpg".format(user_name, count), face_img)
        else:
#             print("Camera not available, close any other apps using the webcam and try again")
            continue    
    cam.release()
    cv2.destroyAllWindows()
    
    return input_embeddings



def convert_bbox(bbox, small_frame): ## from relative bbox to css order
    real_h, real_w, c = small_frame.shape
    x,y,w,h = bbox
    y1 = 0 if y < 0 else y
    x1 = 0 if x < 0 else x 
    y2 = real_h if y1 + h > real_h else y + h
    x2 = real_w if x1 + w > real_w else x + w
    return (y1, x2, y2, x1)