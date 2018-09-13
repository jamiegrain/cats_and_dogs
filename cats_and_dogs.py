import tensorflow as tf 
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
import pickle
from tensorflow.python.keras.callbacks import TensorBoard
import time

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))

NAME = 'cats-vs-dogs-cnn-{}'.format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = tf.keras.utils.normalize(X, axis = 1)

model = tf.keras.models.Sequential()
model.add(Conv2D(64, (3,3), activation = tf.nn.relu, input_shape = X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation = tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64, activation = tf.nn.relu))

model.add(Dense(1, activation = tf.nn.sigmoid))

model.compile(loss = 'binary_crossentropy',
	optimizer = 'adam',
	metrics = ['accuracy'])

model.fit(X, y, epochs = 10, batch_size = 32, validation_split = 0.1, callbacks = [tensorboard])

model.save('cd_model.h5')