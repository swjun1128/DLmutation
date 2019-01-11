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
import random
import numpy as np


def getModel(k):
    # CIFAR10 图片数据集
    # 注意要把数据放到/Users/##/.keras/datasets/cifar-10-batches-py目录下
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32×32

    #X_train 的shape 是32*32*3 所以如果按照modelA和modelB的方法，就在对于32*32的每一个值在进行随机化
    # 噪声
    # 随机抽取1%数据 600
    randomIndex = []
    for i in range(500):
        randomIndex.append(random.randint(0, 49999))
    print(randomIndex)
    # 噪声扰动
    with open("modelC\C" + str(k) + "\\" + "randomResult_" + str(k) + ".txt", "a+", encoding="utf-8") as f:
        for i in range(500):
            f.write(str(randomIndex[i]))
            f.write("\t")
        f.write("\n")
        f.close()

    #获取中间结果
    tempResult = np.zeros((500,32,32,3),np.uint8)
    for i in range(500):
        tempResult[i] = X_train[randomIndex[i]]
    with open("modelC\C"+str(k)+"\\"+"np_"+str(k)+".txt","a+",encoding="utf-8") as f:
        for i in range(500):
            for j in range(32):
                np.savetxt(f,X_train[randomIndex[i]][j])
                f.write("\n")
            f.write("-----------------------------------------------------------------------")
            f.write("\n")
        f.close()

    tempCenterSet = []
    for i in range(500):
        tempCenter = []
        x_axis = random.randint(1, 30)
        y_axis = random.randint(1, 30)
        #观察数据，没有那么多的0
        # while X_train[randomIndex[i]][x_axis][y_axis] == 0:
        #     print("重新随机")
        #     x_axis = random.randint(1, 31)
        #     y_axis = random.randint(1, 31)
        print(x_axis)
        print(y_axis)
        print(X_train[randomIndex[i]][int(x_axis)][int(y_axis)])
        print("\n")
        tempCenter.append(x_axis)
        tempCenter.append(y_axis)
        tempCenterSet.append(tempCenter)
    with open("modelC\C" + str(k) + "\\" + "randomResult_" + str(k) + ".txt", "a+", encoding="utf-8") as f:
        for i in range(500):
            f.write(str(tempCenterSet[i]))
            f.write("\t")
        f.write("\t")
        f.write("\n")
        f.close()

    for i in range(500):
        npArray = np.random.randint(0, 255, (3, 3, 3), np.uint8)
        with open("modelC\C"+str(k)+"\\"+"npArray_"+str(k)+".txt","a+",encoding="utf-8") as arrayF:
            for i in range(3):
                np.savetxt(arrayF,npArray[i])
                arrayF.write("\n")
            arrayF.write("--------------------------------------------------------")
            arrayF.write("\n")
            arrayF.close()
        tempIndex = randomIndex[i]
        tempX = tempCenterSet[i][0]
        tempY = tempCenterSet[i][1]
        # 为了验证随机扰动成功，存储原有的值
        initialArray = np.zeros((3, 3, 3), np.uint8)
        initialArray[0][0] = X_train[tempIndex][int(tempX) - 1][int(tempY) - 1]
        initialArray[0][1] = X_train[tempIndex][int(tempX) - 1][int(tempY)]
        initialArray[0][2] = X_train[tempIndex][int(tempX) - 1][int(tempY) + 1]
        initialArray[1][0] = X_train[tempIndex][int(tempX)][int(tempY) - 1]
        initialArray[1][1] = X_train[tempIndex][int(tempX)][int(tempY)]
        initialArray[1][2] = X_train[tempIndex][int(tempX)][int(tempY) + 1]
        initialArray[2][0] = X_train[tempIndex][int(tempX) + 1][int(tempY) - 1]
        initialArray[2][1] = X_train[tempIndex][int(tempX) + 1][int(tempY)]
        initialArray[2][2] = X_train[tempIndex][int(tempX) + 1][int(tempY) + 1]
        with open("modelC\C"+str(k)+"\\"+"initialArray_"+str(k)+".txt","a+",encoding="utf-8") as initialF:
            for i in range(3):
                np.savetxt(initialF,initialArray[i])
                initialF.write("\n")
            initialF.write("--------------------------------------------------------")
            initialF.write("\n")
            initialF.write("\n")
            initialF.write("\n")
            initialF.close()

            # 变异--噪声扰动
        X_train[tempIndex][int(tempX) - 1][int(tempY) - 1] = npArray[0][0]
        X_train[tempIndex][int(tempX) - 1][int(tempY)] = npArray[0][1]
        X_train[tempIndex][int(tempX) - 1][int(tempY) + 1] = npArray[0][2]
        X_train[tempIndex][int(tempX)][int(tempY) - 1] = npArray[1][0]
        X_train[tempIndex][int(tempX)][int(tempY)] = npArray[1][1]
        X_train[tempIndex][int(tempX)][int(tempY) + 1] = npArray[1][2]
        X_train[tempIndex][int(tempX) + 1][int(tempY) - 1] = npArray[2][0]
        X_train[tempIndex][int(tempX) + 1][int(tempY)] = npArray[2][1]
        X_train[tempIndex][int(tempX) + 1][int(tempY) + 1] = npArray[2][2]

        # 存储中间结果至npedArray.txt
        npedArray = np.zeros((3, 3, 3), np.uint8)
        npedArray[0][0] = X_train[tempIndex][int(tempX) - 1][int(tempY) - 1]
        npedArray[0][1] = X_train[tempIndex][int(tempX) - 1][int(tempY)]
        npedArray[0][2] = X_train[tempIndex][int(tempX) - 1][int(tempY) + 1]
        npedArray[1][0] = X_train[tempIndex][int(tempX)][int(tempY) - 1]
        npedArray[1][1] = X_train[tempIndex][int(tempX)][int(tempY)]
        npedArray[1][2] = X_train[tempIndex][int(tempX)][int(tempY) + 1]
        npedArray[2][0] = X_train[tempIndex][int(tempX) + 1][int(tempY) - 1]
        npedArray[2][1] = X_train[tempIndex][int(tempX) + 1][int(tempY)]
        npedArray[2][2] = X_train[tempIndex][int(tempX) + 1][int(tempY) + 1]
        with open("modelC\C" + str(k) + "\\" + "npedArray_" + str(k) + ".txt", "a+", encoding="utf-8") as npedF:
            for i in range(3):
                np.savetxt(npedF, npedArray[i])
                npedF.write("\n")
            npedF.write("----------------------------------------------------------")
            npedF.write("\n")
            npedF.write("\n")
            npedF.write("\n")
            npedF.close()

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
    with open('modelC\\'+'ModelCResults.txt',"a+",encoding="utf-8") as result_f :
        result_f.write('Test Loss: %.4f' % score[0])
        result_f.write("\t")
        result_f.write('Test accuracy: %.4f'% score[1])
        result_f.write("\n")
        result_f.close()
    model.save('modelC\\'+'ModelC_NP'+str(k)+'.hdf5')

for i in range(10):
    getModel(i)