import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.models import load_model

from keras import backend as K

import os

from PIL import Image


parameters = {
	"batch_size": 32, #32#4#32#128 # ? 32 o mas o que?
	"nb_epoch": 200, #200 # TODO: 200
	"inner_size": 32,
	"outer_size": 48, # == identity / 64
	"window_size": 64
}
parameters_serial = "-".join(["%s_%d" % (k, parameters[k]) for k in parameters])

caltech_dir = "/media/axelbrz/Tera/datasets/101_ObjectCategories"

image_filenames = []
images = []
for category in os.listdir(caltech_dir):
	category_dir = os.path.join(caltech_dir, category)
	for image_filename in os.listdir(category_dir):
		image_path = os.path.join(category_dir, image_filename)
		
		image = Image.open(image_path)
		
		np_image = np.array(image)
		
		# np_image.shape (rows, cols, channels) == (height, width, channels)
		
		images.append(np_image)
		image.close()

np.random.shuffle(images)

images = images[:len(images)/2]


images_training = images[:int(len(images)*.9)]
images_testing = images[int(len(images)*.9):]

def get_samples_from_image(image, parameters):
	inner_size = parameters["inner_size"]
	outer_size = parameters["outer_size"]
	window_size = parameters["window_size"]
	border_size = (outer_size - inner_size) / 2
	
	input_samples = []
	output_samples = []
	
	for y in xrange(0, image.shape[0] - outer_size, window_size):
		for x in xrange(0, image.shape[1] - outer_size, window_size):
			inner_sample = image[y+border_size:y+border_size+inner_size,x+border_size:x+border_size+inner_size]
			outer_sample = image[y:y+outer_size,x:x+outer_size]
			
			input_samples.append(inner_sample)
			output_samples.append(outer_sample)
			
	return input_samples, output_samples

def get_samples_from_images(images, parameters):
	input_all_samples = []
	output_all_samples = []
	for image in images:
		if len(image.shape) == 2 or image.shape[2] == 1:
			continue
		input_samples, output_samples = get_samples_from_image(image, parameters)
		input_all_samples.extend(input_samples)
		output_all_samples.extend(output_samples)
	return input_all_samples, output_all_samples

X_train, Y_train = get_samples_from_images(images_training, parameters)
X_test, Y_test = get_samples_from_images(images_testing, parameters)
"""
for i in xrange(len(X_train)):
	if np.array(X_train[i]).shape != (32, 32, 3):
		print "WHAAAAT"
		exit()

"""

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

train = [(X_train[i], Y_train[i]) for i in xrange(len(X_train))]
test = [(X_test[i], Y_test[i]) for i in xrange(len(X_test))]

np.random.shuffle(train)
np.random.shuffle(test)

X_train = np.array(map(lambda sample: sample[0], train))
Y_train = np.array(map(lambda sample: sample[1], train))
X_test = np.array(map(lambda sample: sample[0], test))
Y_test = np.array(map(lambda sample: sample[1], test))


print len(X_train), len(Y_train)
print len(X_test), len(Y_test)
print len(images)


model_filename = "models/model_zoomouter_%s" % parameters_serial

print "Using model:", model_filename



# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')





input_shape = X_train.shape[1:]
output_shape = Y_train.shape[1:]
print(input_shape)
print(output_shape)


X_train = X_train.astype('float32')
Y_train = Y_train.astype('float32')
X_test = X_test.astype('float32')
Y_test = Y_test.astype('float32')
X_train /= 255
Y_train /= 255
X_test /= 255
Y_test /= 255


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
	
	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	
	model.add(Flatten())
	model.add(Dense(output_shape[0]*output_shape[1]*output_shape[2]))
	model.add(Activation('relu'))
	model.add(Dense(output_shape[0]*output_shape[1]*output_shape[2]))
	model.add(Activation('sigmoid'))
	# model.add(Dropout(0.5))

	model.add(Reshape(output_shape))

	# model.add(Dense(nb_classes))
	# model.add(Activation('softmax'))

	model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])

	model.fit(
		X_train, Y_train, verbose=1,
		batch_size=parameters["batch_size"],
		nb_epoch=parameters["nb_epoch"],
		validation_data=(X_test, Y_test)
	)
	return model

generate_always = True

if generate_always or not os.path.isfile(model_filename):
	model = generate_model()
	model.save(model_filename)
else:
	model = load_model(model_filename)
	
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(score)


def save_tensor_as_image(tensor, filename):
	#print tensor.shape
	rgbArray = np.zeros(tensor.shape, 'uint8')
	rgbArray[..., 0] = tensor[...,0]*255
	rgbArray[..., 1] = tensor[...,1]*255
	rgbArray[..., 2] = tensor[...,2]*255
	img = Image.fromarray(rgbArray)
	#img = np.rot90(np.flipud(img_pixels.reshape(3, 32, 32).T), -1)
	img.save("outputs/%s" % filename, "JPEG")

print "Saving predicted images ..."
for i in xrange(100):
	save_tensor_as_image(X_test[i], "%03d.0.input.jpg" % i)
	save_tensor_as_image(Y_test[i], "%03d.1.expected.jpg" % i)
	output = model.predict(np.array([X_test[i]]), batch_size=1, verbose=0)
	save_tensor_as_image(output[0], "%03d.2.output.jpg" % i)







