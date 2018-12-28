#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:42:34 2018

@author: qq
"""
import sys
sys.path.append('../')
import keras
from keras.models import Model,Input,load_model
from keras.models import load_model
from keras.models import model_from_json
import h5py  #导入工具包  
import numpy as np  
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from boundary import get_bound_data_mnist
from boundary import accuracy_in_bound_data
#mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
'''
#HDF5的写入：  
imgData = np.zeros((30,3,128,256))  
f = h5py.File('test.h5','w')   #创建一个h5文件，文件指针是f  
f['data'] = imgData                 #将数据写入文件的主键data下面  
f['labels'] = range(100)            #将数据写入文件的主键labels下面  
f.close()                           #关闭文件  
   
#HDF5的读取：  
f = h5py.File('test.h5','r')   #打开h5文件  
print f.keys()                            #可以查看所有的主键  ：在这里是：【data】,[label]
print f['labels'][34]
print f['data'][:]               #取出主键为data的所有的键值  
f.close() 
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 输入数据为 mnist 数据集
x_train = x_train.astype('float32').reshape(-1,28,28,1)
x_test = x_test.astype('float32').reshape(-1,28,28,1)

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#print f['/model_weights/conv2d_2']['conv2d_2']['kernel:0'][2][2]

   
model_path='ModelB_raw.hdf5'
model=load_model(model_path)

#参数2是阈值，第一和第二大输出的比
bound_data_lst = get_bound_data_mnist(model,2)
print accuracy_in_bound_data(model,bound_data_lst)
    #print(f['model_weights'][key].value)
