
#This document shows how to build a simple neural network from this MNIST dataset. It utilizes the built-in datasets of Keras to load the data in directly.
# We use only a single convolutional layer followed be a single max pooling layer.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from keras import Sequential

from keras.layers import Dense, Activation, Conv2D, MaxPool2D

#Load the dataset directly from Keras.
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# We will use a callback to cancel training when we have more than 99.8% accuracy on the training data.

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={ }):
        if(logs.get('accuracy')>0.998):
            print("Reached 99.8% accuracy so cancelling training!")
            self.model.stop_training=True
            
callbacks=myCallback()
 
#Reshape in suitable form for convolutions. Scale pixel intensity by max intensity of 255.
X_train=X_train.reshape(60000,28,28,1)
X_train=X_train/255.0


X_test=X_test.reshape(10000,28,28,1)
X_test=X_test/255.0


model= tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),input_shape=(28,28,1),activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'), 
    tf.keras.layers.Dense(10,activation='softmax')
    ])

# Adam Optimization, train for 50 epochs at most
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
my_model= model.fit(X_train,Y_train,epochs=50,callbacks=[callbacks])


