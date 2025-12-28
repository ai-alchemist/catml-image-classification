"""
This program is for using neural nets to classify images.
This file does just inference, not training.
"""

import os
import argparse

import numpy as np
import keras
from keras import layers, ops
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

########################################

####################
# Parsing Arguments
####################

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='model_files',
                    help='Directory to load the model from.')
parser.add_argument('image_path',
                    help='Image to perform inference on.')
args = parser.parse_args()

#assign local variables values from cmdline args.
model_dir = args.model_dir
demo_image_path = args.image_path

# print the arguments.
print("####ARGUMENTS####")
print("Model Directory: " + model_dir)
print("Image Path: " + demo_image_path)

image_size = (180, 180)

########################################

####################
# Prediction
####################

# Load the model from the file.
model = keras.models.load_model(model_dir+"/"+'neural_net.keras')

# Load the image to classify.
img = keras.utils.load_img(demo_image_path, target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis.

# Predictions with support for multiple class types.
predictions = model.predict(img_array)

# Print predictions in decimal, not scientific notation.
np.set_printoptions(suppress=True)

# This prints the confidence that the image contains each class.
# It is up to the user to interpret the values as the relevant
# object types.
print("Predictions for " + demo_image_path + ": " + str(predictions))
