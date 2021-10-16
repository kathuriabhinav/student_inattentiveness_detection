from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def mouth_aspect_ratio(mouth):
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55
	mar = (A + B) / (2.0 * C)
	return mar

MOUTH_AR_THRESH = 0.7
(mStart, mEnd) = (49, 68)

def main_open_mouth(frame, shape):
    mouth = shape[mStart:mEnd]
    mouthMAR = mouth_aspect_ratio(mouth)
    mar = mouthMAR
    mouthHull = cv2.convexHull(mouth)
    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
    cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    if mar >= MOUTH_AR_THRESH:
        cv2.putText(frame, "Mouth is Open!", (30,120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
    elif mar < MOUTH_AR_THRESH:
        cv2.putText(frame, "Mouth is Close!", (30,120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
    return frame