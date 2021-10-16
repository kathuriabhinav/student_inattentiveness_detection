# ivp-online-proctoring-system
IVP mini project

Dependencies: (command to install : pip install "library")
- face_recognition
- imutils
- mediapipe
- cv2
- dlib

 
Issues:
- Noface count, face_verif and multiple count for each frame happening
- Not drawing multiple boxes if multiple faces detected
- Looking down is being treated as looking up in eyeGazeTracking
- Use of Threading
- Using old version of sklearn(0.19.1) for facespoof_det.py
- Mobile detection
- Warnings in headpose estimation (ang2 = ...., m1-m2..)
- (solved)RAM is not getting cleared after ending proctoring system : Use exit() at end of code.
