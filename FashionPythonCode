import tensorflow as tf
import keras
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models
from keras import Sequential
from tensorflow.keras.layers import Conv2D


from keras.layers import Dense, Activation, MaxPooling2D , Flatten

fashion_mnist=keras.datasets.fashion_mnist #Load the Fashion MNSIT data directly from Keras.

(training_images,training_labels),(test_images,test_labels)= fashion_mnist.load_data()


training_images=training_images.reshape(60000,28,28,1) #Reshape in the format needed to apply convolutions.
training_images=training_images/255.0 #Scale by pixel intensity.

test_images=test_images.reshape(10000,28,28,1)
test_images=test_images/255.0


#Define a callback that early stops training at 98% accuracy on training data.
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={ }):
        if(logs.get('accuracy')>0.98):
            print("accuracy is high so stopping training early")
            self.model.stop_training=True
            
        
callbacks=myCallback()


model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),input_shape=(28,28,1),activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
    ])



#Train using Adam Optimizer and standard categorical loss. Minibatch size of 64.
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_images,training_labels,epochs=200,batch_size=64,callbacks=[callbacks])


