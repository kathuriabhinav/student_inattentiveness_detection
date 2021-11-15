import tensorflow as tf 
import cv2
import numpy as np
from keras.preprocessing import image

model1=tf.keras.models.load_model("models/modelface_network1.h5")
objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

def emotion_recog(faces):
    for face in faces:
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