import cv2
import mediapipe as mp

class Face:
    def __init__(self):
        # Face detection
        self.id = None
        self.confidence = None
        self.bbox = None
        self.origbbox = None
        self.img = None
        # Facial Landmarks
        self.landmarks = None
        self.hland = None
        # Face recognition
        self.aligned_face = None
        self.embedding = None
        self.name = None
        self.distance = None
        self.best_index = None
        # Mouth track
        self.mouth = None
        # Head pose estimation
        self.head = None
        # Spoof detection
        self.spoof = None
        self.spoof_score = None
        # emotion detection
        self.emotion = None
        self.emotion_score = None

# Crop face based on its bounding box
def get_face(frame, bbox):
    real_h, real_w, c = frame.shape
    x,y,w,h = bbox
    y1 = 0 if y < 0 else y
    x1 = 0 if x < 0 else x 
    y2 = real_h if y1 + h > real_h else y + h
    x2 = real_w if x1 + w > real_w else x + w
    face = frame[y1:y2,x1:x2,:]
    return face

def expand_bbox(frame, bbox):
    real_h, real_w, _ = frame.shape
    x,y,w,h = bbox
    z = 1.5
    ch = 0.145*h
    x1,y1 = int(x - w*(z-1)/2) , int(y - h*(z-1)/2 -ch)
    x1, y1 = x1 if x1>=0 else 0, y1 if y1>=0 else 0
    w1, h1 = real_w if int(x1+w*z) > real_w else int(w*z) , real_h if int(y1+h*z) > real_h else int(h*z)
    return x1,y1,w1,h1

def detect_faces(frame, confidence = 0.7):
    """
    Outputs the frame with detected face, alert_bool and cropped face
    """
    faces = None
    mp_face_detection = mp.solutions.face_detection
    
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence= confidence) as face_detector:
        
        # To improve performance, optionally mark the frame as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Face detection:
        results = face_detector.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        
    # Get bboxes for detected faces
    if results.detections:
        faces = []
        for id, detection in enumerate(results.detections):
            face = Face()
            face.id = id
            bbox = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            face.origbbox = bbox
            face.confidence = detection.score
            face.bbox = expand_bbox(frame, face.origbbox)
            face.img = get_face(frame, face.bbox)
            faces.append(face)
        
    return faces