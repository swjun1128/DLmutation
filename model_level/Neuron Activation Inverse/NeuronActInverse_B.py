# -*- coding: utf-8 -*-
import keras
from keras.models import load_model
from keras.models import model_from_json
import h5py
import os
import numpy as np
import pandas as pd
from keras.datasets import cifar10
import csv
from keras.datasets import mnist
from random import shuffle

import sys
sys.path.append('../../')
from boundary import get_bound_data_mnist
from boundary import accuracy_in_bound_data_mnist

def HDF5_structure(data):
    root=data.keys()
    final_path=[]
    data_path=[]
    while True:
        if len(root)==0:
            break
        else:
            for item in root:
                if isinstance(data[item],h5py._hl.dataset.Dataset) or len(data[item].items())==0:
                    root.remove(item)
                    final_path.append(item)
                    if isinstance(data[item],h5py._hl.dataset.Dataset):
                        data_path.append(item)
                else:
                    for sub_item in data[item].items():
                        root.append(os.path.join(item,sub_item[0]))
                    root.remove(item)
    return data_path


def neuron_act_inverse(model,Layer='dense_1',neuron_index=0):
    #Layer = dense_1 or dense_2
    #neuron_index第n个神经元
    json_string=model.to_json()
    model.save_weights('my_model_weight.h5')
    
    with h5py.File('my_model_weight.h5', 'r+') as data:
        data_path=HDF5_structure(data)
        lst=[]
        for path in data_path:
            if Layer not in path.split('/')[0]:
                continue
            if 'bias' in path:
                #把偏移值置为相反数
                data[path][neuron_index]=-data[path][neuron_index]
                continue
            #print path
            #print data[path].shape
            #更新data里的数据，把neuron_index神经元的权重都置为相反数
            for i in range(data[path].shape[0]):
                temp_lst=data[path][i].copy()
                temp_lst[neuron_index]=-temp_lst[neuron_index]
                data[path][i] = temp_lst
    model_change = model_from_json(json_string)
    model_change.load_weights('my_model_weight.h5')
    return model_change

def accuracy_mnist(model):
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
    score = accuracy_mnist(model)
    print('Origin Test accuracy: %.4f'% score)
    
    bound_data_lst = get_bound_data_mnist(model,10)
    acclst =[]
    index = np.random.choice(120,25)
    for i in range(25):
        model_change = neuron_act_inverse(model,Layer = 'dense_1',neuron_index=index[i])
        model_change.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        #print 'Mutated Test accuracy: ',accuracy_cifar(model_change)
        acc= accuracy_in_bound_data_mnist(model_change,bound_data_lst)
        acclst.append(acc)
    print 'Mutated accuracy in bound data(dense1):',[round(i,4) for i in acclst]
