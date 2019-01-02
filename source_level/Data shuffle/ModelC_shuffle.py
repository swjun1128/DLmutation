#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras import Model,Input
from sklearn.model_selection import train_test_split
import sys
import time
import keras
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
import  numpy  as np
import random
# CIFAR10 图片数据集
# 注意要把数据放到/Users/##/.keras/datasets/cifar-10-batches-py目录下
def getModel(k):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32×32


# 记录随机抽取的Index
    randomIndex = []
    for i in range(500):
        randomIndex.append(random.randint(0,49999))
    print(randomIndex)
    f = open("ModelC_randomIndex_Record.txt", "a+")
    for i in randomIndex:
        f.write(str(i) + "   ")
    f.write("\n")

    tempArray_x = np.zeros((500,32,32,3))
    tempArray_y = np.zeros((500,1))

    for i in range(500):
        tempArray_x[i] = X_train[randomIndex[i]]
        tempArray_y[i] = Y_train[randomIndex[i]]
    indices =np.random.permutation(tempArray_x.shape[0])
    rand_data_x = tempArray_x[indices]
    rand_data_y = tempArray_y[indices]

    for i in range(500):
        if (X_train[randomIndex[i]] != rand_data_x[i]).any():
            print("Different Success!!!!")
        X_train[randomIndex[i]] = rand_data_x[i]
        Y_train[randomIndex[i]] = rand_data_y[i]



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
    model.add(Dense(10, activation='softmax'))
    model.summary()


    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=128, epochs=20,verbose=1,shuffle=False) #先不要 shuffle=False
    #Y_pred = model.predict_proba(X_test, verbose=0)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('测试集 score(val_loss): %.4f' % score[0])
    print('测试集 accuracy: %.4f' % score[1])
    f.write('Test Loss: %.4f' % score[0])
    f.write("\t")
    f.write('Test accuracy: %.4f' % score[1])
    f.write("\n")
    f.close()
    model.save("ModelC_RS"+str(k)+".hdf5")


for i in range(10):
    getModel(i)