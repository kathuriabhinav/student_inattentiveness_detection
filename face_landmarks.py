import cv2
import math
import mediapipe as mp
import numpy as np
mp_face_mesh = mp.solutions.face_mesh

def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


# # Extracting Landmark points
# print(results.multi_face_landmarks[0].landmark[0])
# results.multi_face_landmarks[0].landmark[0].x


def get_landmarks(face, face_landmarks):
    landmarks = []
    image_rows, image_cols, _ = face.img.shape
    x,y,_,_ = face.bbox
    for pt in face_landmarks:
        px = normalized_to_pixel_coordinates(pt.x, pt.y, image_cols, image_rows)
        if px:
            px = px[0]+x, px[1]+y, pt.z
            landmarks.append(px)
    return landmarks   

def detect_landmarks(frame, faces, det_conf = 0.7, track_conf = 0.7): #Assuming only one face in each face object
    if not faces:
        return

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True, min_detection_confidence=det_conf, min_tracking_confidence= track_conf) as face_mesh:
        
        for face in faces:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            face.img.flags.writeable = False
            face.img = cv2.cvtColor(face.img, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(face.img)
            if results.multi_face_landmarks:
                face.landmarks = get_landmarks(face, results.multi_face_landmarks[0].landmark)
                x,y,w,h = face.bbox
                f_h, f_w,_ = frame.shape
                face.hland = np.array([( ((lm.x * w)+x)/f_w , ((lm.y * h)+y)/f_h , lm.z) for lm in results.multi_face_landmarks[0].landmark])
            
            face.img.flags.writeable = True
            face.img = cv2.cvtColor(face.img, cv2.COLOR_RGB2BGR)