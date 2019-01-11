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
# CIFAR10 图片数据集
# 注意要把数据放到/Users/##/.keras/datasets/cifar-10-batches-py目录下
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32×32

# with open("test.txt","w",encoding="utf-8") as f :
#     np.savetxt(f,X_train[1])
#     f.close()
#噪声
# #随机抽取1%数据 600
# randomIndex = []
# for i in range(600):
#     randomIndex.append(random.randint(0,59999))
# print(randomIndex)
#噪声扰动
# with open("np.txt","a+",encoding="utf-8") as f:
#     for i in range(600):
#         np.savetxt(f,x_train[randomIndex[i]])
#         f.write("\n")
#         f.write("\n")
#         f.write("\n")
#     f.close()
# with open("temp.txt","a+",encoding="utf-8") as f :
#     for i in range(600):
#         f.write(str(randomIndex[i]))
#         f.write("\t")
#     f.write("\n")
#     f.close()

# print(type(x_train[12][2][2]))
# lisy = [[1,2],[2,3]]
# with open("test.txt","a+",encoding="utf-8") as f :
#     f.write(str(lisy))
#
# temp = np.random.randint(0,255,(3,3),np.uint8)
# print(temp)
# with open("npArray.txt","a+",encoding="utf-8") as f:
#     np.savetxt(f,temp)
#     f.close()

# print(np.zeros((3,3),np.uint8))
# with open("modelA\A0\\test.txt","w",encoding="utf-8") as f:
#     f.write("hanyuaneli ")
#     f.close()
x = np.zeros((3,3,3),np.uint8)
x[0][0] = np.random.randint(1,5,(3),np.uint8)
print(x)