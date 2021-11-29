import numpy as np
import math
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def align_faces(faces):
    if not faces:
        return 
    
    for face in faces:
        if not face.landmarks:
            # print("From align_face(): No face landmarks for face-id: {}".format(face.id))
            continue
    
        img = face.img.copy()

        try:
            left_eye = face.landmarks[468][0], face.landmarks[468][1]
            right_eye = face.landmarks[473][0], face.landmarks[473][1]
        except:
            # print("From align_face(): No iris landmarks for face-id: {}".format(face.id))
            continue

        #this function aligns given face in img based on left and right eye coordinates
        left_eye_x, left_eye_y = face.landmarks[468][0], face.landmarks[468][1]
        right_eye_x, right_eye_y = face.landmarks[473][0], face.landmarks[473][1]
        #-----------------------
        #find rotation direction
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 #rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 #rotate inverse direction of clock
        #----------------------
        #find length of triangle edges
        a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
        b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
        c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))
        #-----------------------
        #apply cosine rule
        if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
            cos_a = (b*b + c*c - a*a)/(2*b*c)
            angle = np.arccos(cos_a) #angle in radian
            angle = (angle * 180) / math.pi #radian to degree
            #-----------------------
            #rotate base image
            if direction == -1:
                angle = 90 - angle
                img = Image.fromarray(img)
                img = np.array(img.rotate(direction * angle))
        #-----------------------
        face.aligned_face = img   #return img anyway

        return


def normalize_input(img, normalization = 'base'):

	#issue 131 declares that some normalization techniques improves the accuracy

	if normalization == 'base':
		return img
	else:
		#@trevorgribble and @davedgd contributed this feature

		img *= 255 #restore input in scale of [0, 255] because it was normalized in scale of  [0, 1] in preprocess_face

		if normalization == 'raw':
			pass #return just restored pixels

		elif normalization == 'Facenet':
			mean, std = img.mean(), img.std()
			img = (img - mean) / std

		elif(normalization=="Facenet2018"):
			# simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
			img /= 127.5
			img -= 1

		elif normalization == 'VGGFace':
			# mean subtraction based on VGGFace1 training data
			img[..., 0] -= 93.5940
			img[..., 1] -= 104.7624
			img[..., 2] -= 129.1863

		elif(normalization == 'VGGFace2'):
			# mean subtraction based on VGGFace2 training data
			img[..., 0] -= 91.4953
			img[..., 1] -= 103.8827
			img[..., 2] -= 131.0912

		elif(normalization == 'ArcFace'):
			#Reference study: The faces are cropped and resized to 112×112,
			#and each pixel (ranged between [0, 255]) in RGB images is normalised
			#by subtracting 127.5 then divided by 128.
			img -= 127.5
			img /= 128

	#-----------------------------

	return img


def preprocess_img(img, target_size): 
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)
    dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    img = cv2.resize(img, dsize)
    # Then pad the other side to the target size by adding black pixels
    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]
    # Put the base image in the middle of the padded image
    img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
    #double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)
    #normalizing the image pixels
    img_pixels = image.img_to_array(img) #what this line doing? must?
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255 #normalize input in [0, 1]
    #custom normalization
    img = normalize_input(img = img_pixels, normalization = "Facenet2018")
    return img

def find_input_shape(model):
	# face recognition models have different size of inputs
    #my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
	input_shape = model.layers[0].input_shape
	if type(input_shape) == list:
		input_shape = input_shape[0][1:3]
	else:
		input_shape = input_shape[1:3]
	#----------------------
	#issue 289: it seems that tf 2.5 expects you to resize images with (x, y)
	#whereas its older versions expect (y, x)
	if tf_major_version == 2 and tf_minor_version >= 5:
		x = input_shape[0]; y = input_shape[1]
		input_shape = (y, x)
	#----------------------
	if type(input_shape) == list: #issue 197: some people got array here instead of tuple
		input_shape = tuple(input_shape)

	return input_shape

def represent(faces, model):
    if not faces:
        return 

    for face in faces:
        img = None
        if type(face.aligned_face) is np.ndarray:
            img = face.aligned_face.copy()
        else:
            img = face.img.copy()
        
        # preprocess
        ## decide input shape
        input_shape_x, input_shape_y = find_input_shape(model)
        img = preprocess_img(img, (input_shape_y, input_shape_x))
        
        #represent
        face.embedding = model.predict(img)[0].tolist()

    return

def verify(faces, input_embeddings, metric = "cosine", model_name = "facenet512"):
    if not faces:
        return
        
    # This is original
    # "arcface" : {'cosine': 0.68, 'euclidean': 4.15, 'euclidean_l2': 1.13}
    # 'facenet512':  {'cosine': 0.30, 'euclidean': 23.56, 'euclidean_l2': 1.04},
    thresholds_dict = {
        "arcface": {'cosine': 0.78, 'euclidean': 4.15, 'euclidean_l2': 1.2},
        'facenet512':  {'cosine': 0.39, 'euclidean': 23.56, 'euclidean_l2': 1.04},
    }
    threshold = thresholds_dict[model_name][metric]
    distance = None

    for face in faces:
        minimum_distance = 999 # just initiating min variable
        best_img_index = 0
        for i, input_embed in enumerate(input_embeddings):
            if metric == 'cosine':
                distance = findCosineDistance(face.embedding, input_embed)
            elif metric == 'euclidean':
                distance = findEuclideanDistance(face.embedding, input_embed)
            elif metric == 'euclidean_l2':
                distance = findEuclideanDistance(l2_normalize(face.embedding), l2_normalize(input_embed))
            distance = np.float64(distance)
            if distance < minimum_distance:
                minimum_distance = distance
                best_img_index = i

        if distance <= threshold:
            face.name = "Verified"
        else:
            face.name = "Unknown"

        face.distance = distance
        face.best_index = best_img_index
        
    return

def verify_faces(faces, frmodel, input_embeddings = None):
    align_faces(faces)
    represent(faces, frmodel)
    if input_embeddings:
        verify(faces, input_embeddings, metric = "euclidean_l2")
    return