import cv2
import math
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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


def get_shape(image, face_landmarks):
    shape = {}
    image_rows, image_cols, _ = image.shape
    for indx, pt in enumerate(face_landmarks):
        px = normalized_to_pixel_coordinates(pt.x, pt.y, image_cols, image_rows)
        cv2.circle(image, px, 2, (0, 255, 0), -1)
        shape[indx] = px
    return image, shape   

def face_landmarks_mp(image):
    alert_bool = False
    shape = {}
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.9,
                               min_tracking_confidence= 0.9) as face_mesh:
    
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landmarks:
          alert_bool = True
          for face_landmarks in results.multi_face_landmarks:
              image, shape = get_shape(image, face_landmarks.landmark)
    #         mp_drawing.draw_landmarks(
    #             image=image,
    #             landmark_list=face_landmarks,
    #             connections=mp_face_mesh.FACEMESH_TESSELATION,
    #             landmark_drawing_spec=None,
    #             connection_drawing_spec=mp_drawing_styles
    #             .get_default_face_mesh_tesselation_style())
    #         mp_drawing.draw_landmarks(
    #             image=image,
    #             landmark_list=face_landmarks,
    #             connections=mp_face_mesh.FACEMESH_CONTOURS,
    #             landmark_drawing_spec=None,
    #             connection_drawing_spec=mp_drawing_styles
    #             .get_default_face_mesh_contours_style())
              mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
        return alert_bool, shape, image
        
        
