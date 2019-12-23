
# coding: utf-8

# In[1]:


from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[2]:


from keras.models import Sequential
from keras.layers import Dense

x_train.shape = (60000, 28 * 28)


x_test.shape = (10000, 28 * 28)


x_train = x_train / 255
x_test = x_test / 255

model = Sequential()
model.add(Dense(512,activation='relu',input_dim= 28*28))
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])

from keras.callbacks import TensorBoard

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import keras

keras.backend.set_session(
    tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "localhost:2018"))




hist = model.fit(x_train*255,y_train,batch_size=128,epochs=5,validation_data=(x_test*255,y_test))

