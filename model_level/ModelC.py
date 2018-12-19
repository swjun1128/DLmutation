#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras import Model,Input
import sys
import time
import keras
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

# CIFAR10 图片数据集
# 注意要把数据放到/Users/##/.keras/datasets/cifar-10-batches-py目录下
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32×32



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('Train:{},Test:{}'.format(len(X_train),len(X_test)))

nb_classes=10
# convert integers to dummy variables (one hot encoding)
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
print('data success')


#前面的数据处理没有改。就是按照论文里面的结构搭了一下。

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',padding='valid', input_shape=(32, 32, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',padding='valid'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='linear'))
model.summary()

#categorical_crossentropy
model.compile(loss='mse', optimizer=keras.optimizers.adam(), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=500, epochs=20,verbose=1) #先不要 shuffle=True
#Y_pred = model.predict_proba(X_test, verbose=0)
score = model.evaluate(X_test, Y_test, verbose=0)
print('测试集 score(val_loss): %.4f' % score[0])
print('测试集 accuracy: %.4f' % score[1])
model.save('ModelC_raw.hdf5')
