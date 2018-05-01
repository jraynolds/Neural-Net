"""
This image classifier runs a given set of images through a given neural network file,
then prints whether the neural network thinks the image predicts a cat or a dog.
"""
__author__ = "Jasper Raynolds"
__license__ = "MIT"
__date__ = "April 2018"

import sys
import os
import argparse
import numpy as np
from PIL import Image
from keras.models import load_model
import ntpath

def classify(neural_net, image_file):
	"""
	Using the given model and image file, returns the model's prediction
	for the image as an array.
	"""
	img = Image.open(image_file)
	img.load()
	img_array = np.asarray(img)
	img_array.shape = (1, 100, 100, 3)

	prediction = model.predict(img_array)[0][0]
	return prediction

parser = argparse.ArgumentParser(description='Using a neural network file, classify an image, a set of images, or a directory of images as cats or dogs.')
parser.add_argument("net", help="neural network file to load")
parser.add_argument("image", nargs="+", help="image file, files or directory to classify")
parser.add_argument("-e", action="store_true", help="turn on filename error checking", required=False)
arguments = vars(parser.parse_args())

model = load_model(arguments["net"])

for i in range(len(arguments["image"])):
	files = []
	classifications = []
	corrects = {True:0, False:0}

	# Get all images
	if os.path.isfile(arguments["image"][i]):
		files.append(arguments["image"][i])
	elif os.path.isdir(arguments["image"][i]):
		for file in os.listdir(arguments["image"][i]):
			if file.endswith(".jpg"):
				files.append(arguments["image"][i] + "/" + file)
	else :
		print(arguments["image"][i] + " is not a valid file.")
		continue

	# Run each file
	for file in files:
		catness = classify(model, file)
		# We assume our sigmoid neural net thinks 1 is a cat and 0 is a dog.
		if catness > .5:
			classifications.append("cat")
		else :
			classifications.append("dog")

	# Output classifications
	for j in range(len(classifications)):
		string = ntpath.basename(files[j]) + " is a " + classifications[j] + ". "
		firstChar = {"file": ntpath.basename(files[j])[0], "classification": classifications[j][0]}
		correct = firstChar["file"] == firstChar["classification"]
		if arguments["e"]:
			if firstChar["file"] == "c" or firstChar["file"] == "d":
				string += str(correct)
				corrects[correct] += 1
		print(string)

	# Output our total performance
	if arguments["e"]:
		print("total performance: {0} correct guesses, {1} incorrect guesses. {2}% right.".format(corrects[True], corrects[False], corrects[True]/(corrects[True] + corrects[False])*100))