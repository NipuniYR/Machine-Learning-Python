# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 11:41:50 2021

@author: Nipuni
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import datetime

mnist = datasets.mnist

# MNIST dataset - clean and perfectly preprocessed dataset
# grey scale images (only 1 channel) of same size (28 * 28 pixels images)
# 10,000 test images
# 60,000 train images

# Load data
(x_train, y_train) , (x_test, y_test) = mnist.load_data()

# Rescaling pixel values from 0 -> 255 to 0 -> 1
x_train, x_test = x_train/255.0, x_test/255.0

# Reshaping the input dataset to be able to pass into model.fit
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Names of the 10 classes
target_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# ####################### Creating the convolution architecture ########################
# Model object
model = models.Sequential()

# Covolution layer 1
model.add(layers.Conv2D(
    #number of kernels
    filters=32,
    #size of one kernel
    kernel_size=(3,3),
    #kernal slides 1 pixel vertically or 1 pixel horizanotally at a time
    strides=(1,1),
    #add padding in a way that the size of the input and the output image will be the same
    #output feature map size will be (28,28) (same as input image size)
    padding='same',
    #Relu activation function will be used in this layer
    activation='relu',
    #input image is a 28,28 image with only 1 chanel (as it's a grey scale image)
    input_shape=(28,28,1)))

# Pooling layer 1
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Convolution layer 2
model.add(layers.Conv2D(
    filters=64,
    kernel_size=(3,3),
    strides=(1,1),
    padding='same',
    activation='relu'))

# Pooling layer 2
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Convolution layer 3
model.add(layers.Conv2D(
    filters=64,
    kernel_size=(3,3),
    strides=(1,1),
    padding='same',
    activation='relu'))

# Flatten the output before passig as the input into classification layers
model.add(layers.Flatten())

# Dropout layer - to reduce overfitting
model.add(layers.Dropout(rate=0.3))
# Only 70% of the neurons in the classification layer will be trained everytime
# The rest 30% will be turned off

# Classification layers (Fully connected layers)
model.add(layers.Dense(units=64, activation='relu'))

model.add(layers.Dropout(rate=0.5))

# Final layer (fully connected)
model.add(layers.Dense(units=10, activation='softmax'))

model.summary()
# ####################### End of the convolution architecture ########################

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #metrics is a list of metrics to be evaluated
              #metrics=['accuracy'] calculates how often predictions equal labels
              metrics=['accuracy'])

# Trains the model for 10 iterations (epochs)
history = model.fit(x_train, 
                    y_train, 
                    epochs=3, 
                    validation_data=(x_test, y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# Evaluate test accuracy
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
# Verbose - How the output while training is in progress will show in the console
# 0-show nothing 1-======(animated) 2-Epoch 1/10 

# Generating output predictions for test data set
prediction  = model.predict(x_test)
predictionDF = pd.DataFrame(data=prediction, columns=target_labels)
# Let's select a random index to get a random test image
i=random.randint(0,9999)
print ("Predicted probabilities for the selected image:")
print(predictionDF.iloc[i])
print("Actual label of the image at index ", i, ": ", y_test[i])
print ("Predicted label of the selected image: " , predictionDF.iloc[i].idxmax())

# Few actual and predicted results with images
for i in range(5):
    j = random.randint(0,9999)
    plt.grid(False)
    plt.figure(figsize= (4,4))
    plt.imshow(x_test[j],cmap=plt.cm.binary)
    plt.xlabel("Actual : " + str(y_test[j]))
    plt.title("Predicted :" +  str(predictionDF.iloc[j].idxmax()))
plt.show()