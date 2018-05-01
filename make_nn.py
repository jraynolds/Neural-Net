"""
This neural network creator loads images from a given unsorted directory and creates,
then saves the network to the filename provided.
It uses no testing set.
"""
__author__ = "Jasper Raynolds"
__license__ = "MIT"
__date__ = "April 2018"

import sys
import os
import numpy as np
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten	
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from keras.layers import LeakyReLU

def load_images(image_dir):
	"""
	Given an unsorted directory of image files, returns a tuple of a numpy array of those images 
	converted to shape (100, 100, 3), and a list of 0s and 1s.
	We assume that dog pictures have filenames beginning with "d", and cat pictures "c". 
	Dogs are output as "0" and cats as "1".
	"""
	files = os.listdir(image_dir)
	images = np.zeros(shape=(len(files), 100, 100, 3), dtype = "float16")
	labels = []

	for i in range(len(files)):
		img = Image.open(image_dir + "/" + files[i])
		img.load()
		img_array = np.atleast_3d(np.asarray(img))
		images[i] = (img_array)
		if files[i][0] == "d":
			labels.append(0)
		else :
			labels.append(1)

	labels = np.array(labels)

	return (images, labels)

def create_model(image_dir):
	"""
	Returns a convolutional neural network. Sends the given image directory
	to be loaded, then trains through it.
	Note there is no testing process. This is because we assume the volume of the 
	training set is massive enough to reduce the need for it.
	"""
	images = load_images(image_dir)
	x_train = images[0]
	y_train = images[1]
	print(x_train.shape)
	print(y_train.shape)

	IMAGE_WIDTH, IMAGE_HEIGHT = 100, 100
	EPOCHS = 50
	BATCH_SIZE = 16
	# FINAL_ACTIVATION = "softmax"
	FINAL_ACTIVATION = "sigmoid"
	# OPTIMIZER = keras.optimizers.Adamax()
	OPTIMIZER = "adam"
	STRIDES = (2, 2)
	# DROPOUT = 0.25
	DROPOUT = 0.2

	INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape = INPUT_SHAPE))
	model.add(keras.layers.LeakyReLU())
	model.add(MaxPooling2D(pool_size = (2, 2), strides = STRIDES))

	model.add(Conv2D(32, (3, 3)))
	model.add(keras.layers.LeakyReLU())
	model.add(MaxPooling2D(pool_size = (2, 2), strides = STRIDES))

	model.add(Conv2D(64, (3, 3)))
	model.add(keras.layers.LeakyReLU())
	model.add(MaxPooling2D(pool_size = (2, 2), strides = STRIDES))

	# model.add(Conv2D(64, (3, 3)))
	# model.add(keras.layers.LeakyReLU())
	# model.add(MaxPooling2D(pool_size = (2, 2), strides = STRIDES))

	model.add(Flatten())

	# model.add(Dense(100)) # 100, 50, 2 recommended
	model.add(Dense(64)) # 100, 50, 2 recommended
	model.add(keras.layers.LeakyReLU())
	model.add(Dropout(DROPOUT))

	# model.add(Dense(50))
	# model.add(keras.layers.LeakyReLU())
	# model.add(Dropout(DROPOUT))

	# model.add(Dense(2))
	model.add(Dense(1))
	model.add(Activation(FINAL_ACTIVATION))

	model.compile(
		loss = "binary_crossentropy",
		# loss = "sparse_categorical_crossentropy",
		optimizer = OPTIMIZER,
		metrics = ["accuracy"]
	)

	# We modify the training set by distorting the images slightly.
	train_datagen = ImageDataGenerator(
		rescale = 1./255,
		shear_range = 0.2,
		zoom_range = 0.2,
		horizontal_flip = True
	)

	train_generator = train_datagen.flow(
		x_train,
		y_train,
		batch_size = BATCH_SIZE
	)

	# model.summary()

	model.fit_generator(
		train_generator,
		steps_per_epoch = x_train.shape[0] // BATCH_SIZE,
		epochs = EPOCHS
	)

	return model

arguments = sys.argv
assert len(arguments) == 3, "You must provide two arguments: an image directory and the name of the output file."
create_model(arguments[1]).save(arguments[2], include_optimizer = False)