#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras import Model,Input
from sklearn.model_selection import train_test_split
import sys
import time
import random
import numpy as np
import keras
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

# CIFAR10 图片数据集
# 注意要把数据放到/Users/##/.keras/datasets/cifar-10-batches-py目录下
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # 32×32
# (50000, 32, 32, 3)
# (50000, 1)

#定义随即复制变量
x_train_re = np.zeros((5000,32,32,3))
y_train_re = np.zeros((5000,1))
# [432, 919, 57, 757, 215, 738, 430, 357, 326, 55]
#制定随机种子（就是上面的seedslist中的值），便于复现
random.seed(55)    #
ranIndexList = []
for i in range(5000):
    temp = random.randint(0,49999)
    ranIndexList.append(temp)
    x_train_re[i] = x_train[temp]
    y_train_re[i] = y_train[temp]
    # print(x_train_re.shape)  (28, 28)
print(ranIndexList)   #打印不同随机种子生成的随机采样结果。
x_train = np.concatenate((x_train_re, x_train), axis=0)
y_train = np.concatenate((y_train_re, y_train), axis=0)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('Train:{},Test:{}'.format(len(x_train),len(x_test)))

nb_classes=10
# convert integers to dummy variables (one hot encoding)
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
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
model.add(Dense(10, activation='softmax'))
model.summary()


model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=15,verbose=1) #先不要 shuffle=True
#Y_pred = model.predict_proba(X_test, verbose=0)
score = model.evaluate(x_test, y_test, verbose=0)
print('测试集 score(val_loss): %.4f' % score[0])
print('测试集 accuracy: %.4f' % score[1])
model.save('ModelC_seed55.hdf5')