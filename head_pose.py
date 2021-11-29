import cv2
import numpy as np
import math
from models.face_geometry import ( 
    PCF,
    get_metric_landmarks
)
font = cv2.FONT_HERSHEY_SIMPLEX 

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    rear_size = 0
    rear_depth = 6
    front_size = 10 
    front_depth = 10
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    # cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[5]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[6]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[7]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[5]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    
    for pt in point_2d:
        cv2.circle(img, tuple(pt), 2, (255,0,0), -1)                   
         

def get_2d_points(rotation_vector, translation_vector, camera_matrix, val):
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def head_pose_points(frame, rotation_vector, translation_vector, camera_matrix, val, dir = "horiz"):
    point_2d = get_2d_points(rotation_vector, translation_vector, camera_matrix, val)
    x,y = None, None
    if dir=="horiz":
        y = tuple((point_2d[5] + point_2d[8])//2)
        x = tuple(point_2d[2])
    elif dir == "vert":
        y = tuple((point_2d[5] + point_2d[6])//2)
        x = tuple(point_2d[2])
    return (x, y)


def headpose_est(frame, faces):
    if not faces:
        return
    
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
        landmarks = face.hland[:468, :].T
        points_idx = [4, 33, 61, 199, 263, 291]
        frame_height, frame_width, channels = frame.shape
        
        # pseudo camera internals
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
            )
        dist_coeffs = np.zeros((4, 1))
        pcf = PCF(
            near=1,
            far=10000,
            frame_height=frame_height,
            frame_width=frame_width,
            fy=camera_matrix[1, 1],
        )

        metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), pcf)
        model_points = metric_landmarks[0:3, points_idx].T
        image_points = (landmarks[0:2, points_idx].T * np.array([frame_width, frame_height])[None, :])
        success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

        
        # x1, p1 = (int(image_points[0][0]), int(image_points[0][1])), (int(image_points[0][0]), int(image_points[0][1]))


        # val = [rear_size, rear_depth, front_size, front_depth]
        # For measuring: val = [0, 0, 15, 15] ; p1, p2, x1, x2
        p1,p2 = head_pose_points(frame, rotation_vector, translation_vector, camera_matrix, val = [0,0,15,15],dir="vert")
        x1, x2 = head_pose_points(frame, rotation_vector, translation_vector, camera_matrix,val = [0,0,15,15],dir="horiz") 

        if (p2[0] - p1[0]) == 0:
            ang1 = 90
        else:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))

        if (x2[0] - x1[0]) == 0:
            ang2 = 90
        else:
            m = (x2[1] - x1[1])/(x2[0] - x1[0])
            ang2 = int(math.degrees(math.atan(-1/m)))
        
        tempstr="Head Straight"
        hdir = None
        vdir = None

        if ang2 >= 20 :
            tempstr="Head left"
        elif ang2 <= -20:
            tempstr="Head right"
        elif ang1 >= 20:
            tempstr="Head up"
        elif ang1 <= -20: #10
            tempstr="Head down"
        else:
            tempstr="Head Straight"

        if ang1<0:
            vdir = "down"
        elif ang1>0:
            vdir = "up"
        else:
            vdir = "up"
        
        if ang2<0:
            hdir = "right"
        elif ang2>0:
            hdir = "left"
        else:
            hdir = "right"
        
        # for drawing: val = [0, 6, 10, 10]  ;  pd1, pd2, xd1, xd2
        draw_annotation_box(frame, rotation_vector, translation_vector, camera_matrix)
        pd1, pd2 = head_pose_points(frame, rotation_vector, translation_vector, camera_matrix, val = [0,6,10,10],dir="vert")
        xd1, xd2 = head_pose_points(frame, rotation_vector, translation_vector, camera_matrix, val = [0,6,10,10],dir="horiz")
        cv2.line(frame, pd1, pd2, (0, 255, 255), 2)
        cv2.line(frame, xd1, xd2, (255, 255, 0), 2)

        # To draw arrowed line
        # pt = get_2d_points(rotation_vector, translation_vector, camera_matrix, [0,6,0,10])
        # cv2.circle(frame, tuple(pt[5]), 4, (0,0,255), -1) 
        # cv2.circle(frame, tuple(pt[1]), 4, (0,0,255), -1)                   
        # frame = cv2.arrowedLine(frame, tuple(pt[1]), tuple(pt[5]), (235, 235, 0), 2, tipLength = 0.2) 

        cv2.putText(frame, str(ang1), pd2, font, 0.7, (128, 255, 255), 2)
        cv2.putText(frame, str(ang2), xd2, font, 0.7, (255, 255, 128), 2)
        
        cv2.putText(frame, tempstr, (15,80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)
        cv2.putText(frame, vdir , (15,105),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)
        cv2.putText(frame, ": "+str(abs(ang1)) , (60,105),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)
        cv2.putText(frame, hdir , (100,105),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)
        cv2.putText(frame, ": "+str(abs(ang2)) , (135,105),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)

        face.head = tempstr
    
    return 