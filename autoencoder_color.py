'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.models import load_model

from keras import backend as K

import os

model_filename = "modelito_color"

batch_size = 128 # ? 32 o mas o que?
nb_classes = 10
nb_epoch = 200 # TODO: 200

# input image dimensions
img_rows, img_cols = 32, 32

img_channels = 3

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
"""
if K.image_dim_ordering() == 'th':
	X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)
"""

input_shape = X_train.shape[1:] #(img_rows, img_cols, img_channels)
print(input_shape)

print("AAAAAAA", X_train[0].shape)

# REC PARA AXEL ==== WARNING, CHECK THIS ==================================================

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

def generate_model():
	model = Sequential()

	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
							border_mode='valid',
							input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(img_rows * img_cols * 3))
	model.add(Activation('sigmoid'))
	# model.add(Dropout(0.5))

	model.add(Reshape(input_shape))

	# model.add(Dense(nb_classes))
	# model.add(Activation('softmax'))

	model.compile(loss='mse',
				  optimizer='adadelta',
				  metrics=['accuracy'])

	model.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,
			  verbose=1, validation_data=(X_test, X_test))
	return model

generate_always = True

if generate_always or not os.path.isfile(model_filename):
	model = generate_model()
	model.save(model_filename)
else:
	model = load_model(model_filename)
	
score = model.evaluate(X_test, X_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(score)


from PIL import Image
def save_tensor_as_image(tensor, filename):
	rgbArray = np.zeros((img_rows, img_cols, 3), 'uint8')
	rgbArray[..., 0] = tensor[...,0]*255
	rgbArray[..., 1] = tensor[...,1]*255
	rgbArray[..., 2] = tensor[...,2]*255
	img = Image.fromarray(rgbArray)
	#img = np.rot90(np.flipud(img_pixels.reshape(3, 32, 32).T), -1)
	img.save(filename, "JPEG")


for i in xrange(10):
	save_tensor_as_image(X_test[i], "%03d.0.input.jpg" % i)
	output = model.predict(np.array([X_test[i]]), batch_size=1, verbose=1)
	save_tensor_as_image(output, "%03d.1.output.jpg" % i)

# TODO: TO DEBUG:
exit()
img_3 = np.array(Image.open("A.jpg"))
img = np.zeros((28,28,1), 'uint8')
img[...,0] = img_3[...,0]
print(img)
img = img.astype('float32')
img /= 255
#save_tensor_as_image(img, "img.jpg")

output_A = model.predict(np.array([img]), batch_size=1, verbose=1)
save_tensor_as_image(output_A, "AAAA.jpg")
