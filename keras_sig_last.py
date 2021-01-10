#/usr/bin/python

import tcga_tools, json, math

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow.keras.backend as K

# reset!
K.clear_session()

# get the config variables
TCGA_root = tcga_tools.file_root
chromosome = tcga_tools.chromosome

##
## INPUTS
##

# Data is stored in two pre-cleaned and pre-sorted numpy arrays with shape (samples, features)
#
# The sample and feature count depends on the preprocessing step, but it stored in a json file, so get that first

array_info_file = open(TCGA_root + 'output/ch_%s_npy_info.json' % chromosome, 'r')
array_info = json.load(array_info_file)
array_info_file.close()

samples = array_info['samples']
input_length = array_info['inputs']
output_length = array_info['outputs']

# Next, get the training data
train_x = np.load(TCGA_root + 'output/meth450_%s.npy' % chromosome)
train_y = np.load(TCGA_root + 'output/mRNA_%s.npy' % chromosome)



##
## THE NEURAL NET ARCHITECTURE
##


# The input layer takes (features, samples) as its input, but samples need not specifically be defined
# This is a 'dummy' layer and only needs the shape information
inputlayer = keras.Input(shape=(input_length,))


# Next, we have some collection of dense layers
# These need the number of nodes and activation, but more can be tuned if necessary
#dense1 = keras.layers.Dense(input_length, activation='relu', kernel_initializer=keras.initializers.RandomNormal, kernel_regularizer=keras.regularizers.l1(0.1))(inputlayer)
dense1 = keras.layers.Dense(input_length, activation='relu', kernel_regularizer=keras.regularizers.l1(0.1))(inputlayer)

#dense2 = keras.layers.Dense(math.floor(input_length/2), activation='relu')(dense1)
#dense3 = keras.layers.Dense(21, activation='relu')(dense2)

# THe output layer must be linear since we are trying to do regression
#outputlayer = keras.layers.Dense(output_length, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal)(dense1)
outputlayer = keras.layers.Dense(output_length, activation='sigmoid')(dense1)


##
## SETTING UP THE MODEL
##

# Using the functional form specified above, wire up the neural net
model = keras.Model(inputs=inputlayer, outputs=outputlayer)

# Print some useful information
model.summary()



##
## FITTING THE MODEL TO THE DATA
##

# Here, we specify what loss function and optimizer to use


model.compile(loss='MSE', optimizer='Adam', metrics=['accuracy'])


model.load_weights('ch_%s_weights' % chromosome)

# Fit the model, using specified options
history = model.fit(train_x, train_y, batch_size=128, epochs=5, validation_split=0.2)


# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



model.save_weights('ch_%s_weights' % chromosome, save_format='tf')

print('waiting')