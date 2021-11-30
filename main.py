import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Face detection
from face_detector import detect_faces
# Face landmarks detection
from face_landmarks import detect_landmarks
# Face recognition
from face_recognition import verify_faces
from models.Facenet512 import loadFaceNet512Model
# Other Face modules
from face_spoofing import spoof_detector
from head_pose import headpose_est
from mouth_detector import mouth_open
from emotion import emotion_recog
# Cheat predicting modules
from cheating_detector import *
from plot_graphs import *
# Utils
import cv2
from utils import register_user, print_fps, print_faces
#sleep
from sleep import sleep_main

sleep_flag = 0
yawn_flag = 0
count_mouth = 0
counter = 0
total = 0
total_yawn = 0
font = cv2.FONT_HERSHEY_SIMPLEX 
pTime = [0]

# Register User
frmodel = loadFaceNet512Model()
input_embeddings, input_im_list = register_user(frmodel, 0)

if __name__ == "__main__":
    
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('output_video.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

    cv2.namedWindow('MONITORING ON')
    frames=[]
    t1 = time.time()
    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret :
            break

        print_fps(frame, pTime)

        faces =  detect_faces(frame, confidence = 0.7)
        detect_landmarks(frame, faces, det_conf = 0.7, track_conf = 0.7)
        verify_faces(faces, frmodel, input_embeddings)
        spoof_detector(faces)        
        headpose_est(frame, faces)
        mouth_open(faces)
        emotion_recog(faces)
        sleep_flag,yawn_flag,count_mouth,counter,total,total_yawn = sleep_main(faces,frame,sleep_flag,yawn_flag,count_mouth,counter,total,total_yawn)

        print_faces(frame, faces)

        cheat_temp = detect_cheating_frame(faces,frames)
        frames.append(cheat_temp)
        if cheat_temp.cheat>0:
            cv2.putText(frame, "INATTENTIVE", (15,150), font, 1, (0,0,255),2)

        cv2.imshow('MONITORING ON',  frame)
        result.write(frame)

        if cv2.waitKey(1) & 0xFF == 27: 
            break

    t2 = time.time()
    if frames:
        fps = int(frames[-1].count/(t2-t1))

    cap.release()
    result.release()
    cv2.destroyAllWindows()


    segments = detect_cheating_segment(frames, fps)
    plot_main(frames, fps)
    plot_segments(segments, [])