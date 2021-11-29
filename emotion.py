import tensorflow as tf 
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

model1=tf.keras.models.load_model("models/modelface_network1.h5")
objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

def emotion_recog(faces):
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
        img= cv2.cvtColor(face.img ,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(48,48))
        arr= image.img_to_array(img)
        arr = np.expand_dims(arr, axis = 0)
        arr/= 255
        custom = model1.predict(arr)

        if objects[np.argmax(custom)]=='happy':
            face.emotion = "happy"
        else:
            face.emotion = "neutral"
        face.emotion_score = 1 - custom[0][3]