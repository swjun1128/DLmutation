#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 19:24:08 2019

@author: qq
"""

import keras
from keras.models import load_model
from keras.models import model_from_json
import h5py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
import csv
from keras.datasets import mnist
import glob

import sys
sys.path.append('../../')
import boundary
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd


def mutant_kill_rate(path,test_suite,mnist):
    image_data=mnist.test.images.astype('float32').reshape(-1,28,28,1)  
    killed=0
    path_lst=glob.glob(path+'/*.hdf5')
    total =len(path_lst)
    for index,path in enumerate(path_lst):
        print index
        mutant=load_model(path)
        pred=mutant.predict(image_data)
        pred=np.array(map(lambda x:np.argmax(x),pred))
        label=np.array(map(lambda x:np.argmax(x),mnist.test.labels))
        for tc in test_suite:
            if pred[tc]!=label[tc]:#结果不一致，说明能杀死
                killed+=1
                print "killed",killed
                break
    return killed,total


def accuracy_mnist(model,mnist):
    '''
    model: DNN_model
    return : acc of mnist
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 输入数据为 mnist 数据集
    x_test = x_test.astype('float32').reshape(-1,28,28,1)
    x_test = x_test / 255

    y_test = keras.utils.to_categorical(y_test, 10)
    score = model.evaluate(x_test, y_test)
    return score[1]


if __name__=='__main__':
    model_path='../ModelB_raw.hdf5'
    model=load_model(model_path)    
    bound_data_lst = boundary.get_bound_data_mnist(model,10)#第一是第二的10倍以内，算边界值
    unbound_data_lst = boundary.get_unbound_data_mnist(model,10)
    
    path_lst=glob.glob('./mutated/MODEL_B/*.hdf5')
    bound_acc_list=[]#变异模型在边界值集合上的准确率
    unbound_acc_list=[]#变异模型在非边界值集合上的准确率
    acc_list=[]
    print('swj')
    for index,path in enumerate(path_lst):
        print index
        mutant_model=load_model(path)
        acc = accuracy_mnist(mutant_model,mnist)
        acc_list.append(acc)
        bound_acc =boundary.accuracy_in_bound_data_mnist(mutant_model,bound_data_lst)
        unbound_acc =boundary.accuracy_in_unbound_data_mnist(mutant_model,unbound_data_lst)
        bound_acc_list.append(bound_acc)
        unbound_acc_list.append(unbound_acc)
        print bound_acc
        print unbound_acc
        
    name=['normal_acc','bound_acc','unbound_acc']
    dataframe = pd.DataFrame({'normal_acc':acc_list,'bound_acc':bound_acc_list,'unbound_acc':unbound_acc_list})
    dataframe.to_csv('result_model_B.csv')