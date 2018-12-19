#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:42:34 2018

@author: qq
"""
import keras
from keras.models import load_model
from keras.models import model_from_json
import h5py  #导入工具包  
import numpy as np  
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from keras.datasets import cifar10
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
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 输入数据为 mnist 数据集
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
   

model_path='../ModelC_raw.hdf5'
model=load_model(model_path)
json_string=model.to_json()
#relu换成linear即可   
#replace('relu', 'linear', 1)
#find('relu')

score = model.evaluate(x_test, y_test)
print('Origin Test accuracy: %.4f'% score[1])

for i in range(json_string.count('relu')):
    #轮流换relu，换成linear其实就是删除的效果，线性等于无激活函数
    json_string_temp = json_string.replace('relu', 'linear', i+1)
    json_string_new = json_string_temp.replace('linear','relu',i)
    print json_string_new.count('relu')
    print json_string_new.find('linear')
    model_change = model_from_json(json_string_new)
    model_change.set_weights(model.get_weights())
    #重新编译
    model_change.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    score = model_change.evaluate(x_test, y_test)
    print('Mutated Test accuracy: %.4f'% score[1])

'''
10000/10000 [==============================] - 3s 295us/step
Mutated Test accuracy: 0.9712
3
1188
10000/10000 [==============================] - 3s 295us/step
Mutated Test accuracy: 0.9651
3
2036
10000/10000 [==============================] - 3s 291us/step
Mutated Test accuracy: 0.8997
3
2503
10000/10000 [==============================] - 3s 293us/step
Mutated Test accuracy: 0.9243
'''
#model.save_weights('my_model_weight.h5')
#data=h5py.File('my_model_weight.h5','r+')
    #print(f['model_weights'][key].shape) 

    #print(f['model_weights'][key].value)
