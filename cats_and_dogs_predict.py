import tensorflow as tf 
import numpy as np 
import os
import matplotlib.pyplot as plt
import cv2
import pickle

IMG_SIZE = 80

predict_test = []
test_images = ['doggo.jpg', 'doggo1.jpg', 'kitty.jpg', 'kitty1.jpg']

def read_test_data():
	# for image in test_images:	
	try:
		img_array = cv2.imread('doggo.jpg', cv2.IMREAD_GRAYSCALE)
		new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
		predict_test.append(new_array)
	except Exception as e:
		pass
			
read_test_data()

model = tf.keras.models.load_model('cd_model.h5')

predict_test = np.array(predict_test)
predict_test = np.reshape(predict_test, (-1, IMG_SIZE, IMG_SIZE, 1))

if model.predict(predict_test) == 1:
	print('Cat')
else:
	print('Doggo')