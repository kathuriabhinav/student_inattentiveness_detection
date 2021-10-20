import cv2
import utils
import numpy as np
from face_detector import detect_faces
from face_recognition import Recognizer
from face_spoofing import spoof_detector
from face_landmarks import detect_landmarks
from head_pose import headpose_est
from mouth_detector import mouth_open
# from eye_tracker import eye_tracking
from cheating_detector import *
from plot_graphs import *

font = cv2.FONT_HERSHEY_SIMPLEX 
pTime = [0]

# Face recognizer
fr = Recognizer(threshold = 0.8)

# Register User
fr.input_embeddings = utils.register_user(fr, num_pics = 5, skipr = False)

if __name__ == "__main__":
    cap = cv2.VideoCapture("inhouse_dataset/abhi.mp4")
    cv2.namedWindow('PROCTORING ON')
    frames=[]
    while(True):

        ret, frame = cap.read()
        if not ret :
            break
        frame = utils.print_fps(cv2.flip(frame, 1), pTime)
        
        faces =  detect_faces(frame, confidence = 0.7)
        if faces:
            fr.verify_faces(faces)
            spoof_detector(faces)
            if len(faces)==1:
                hland = detect_landmarks(frame, faces) 
                if faces[0].landmarks:
                    faces[0].head = headpose_est(frame, faces, hland)
                    # eye_tracking(frame, faces[0].shape, threshold = 75)
                    faces[0].mouth = mouth_open(frame, faces)
            frame = utils.print_faces(frame, faces)
        cheat_temp = detect_cheating_frame(faces,frames)
        frames.append(cheat_temp)
        if cheat_temp.cheat>0:
            cv2.putText(frame, "Cheating suspected", (15,105), font, 0.5, (0,0,255),2)
        cv2.imshow('PROCTORING ON',  frame)
                
        if cv2.waitKey(1) & 0xFF == 27: 
            break
    cap.release()
    cv2.destroyAllWindows()
    plot_main(frames)
    segments = segment_count(frames)
    print_stats(segments)
    plot_segments(segments)