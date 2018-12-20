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


def neuron_switch(model,Layer='dense_1',neuron_change=[0,1]):
    print neuron_change
    #Layer = dense_1 or dense_2
    #neuron_index第n个神经元
    json_string=model.to_json()
    model.save_weights('my_model_weight.h5')
    
    with h5py.File('my_model_weight.h5', 'r+') as data:
        data_path=HDF5_structure(data)
        lst=[]
        for path in data_path:
            #print path
            #print data[path].shape
            next_layer ='dense_3'
            if Layer=='dense_1':
                next_layer = 'dense_2'
                
            if next_layer not in path.split('/')[0] or 'bias' in path:
                continue                
            
            temp_lst=data[path][neuron_change[0]].copy()
            data[path][neuron_change[0]] = data[path][neuron_change[1]].copy()
            data[path][neuron_change[1]] = temp_lst
    model_change = model_from_json(json_string)
    model_change.load_weights('my_model_weight.h5')
    return model_change

def accuracy_cifar(model):
    #model: CNN_model
    #return : acc of cifar
    (_, _), (X_test, Y_test) = cifar10.load_data()
    X_test=X_test.astype('float32')
    X_test/=255
    Y_test = keras.utils.to_categorical(Y_test, 10)
    score = model.evaluate(X_test, Y_test, verbose=0)
    return score[1]


if __name__=='__main__':
    model_path='../ModelC_raw.hdf5'
    model=load_model(model_path)
    score = accuracy_cifar(model)
    print('Origin Test accuracy: %.4f'% score)
    acc =[]
    for i in range(25):
        model_change = neuron_switch(model,Layer = 'dense_1',neuron_change=np.random.choice(256,2))
        model_change.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        #print 'Mutated Test accuracy: ',accuracy_cifar(model_change)
        acc.append(accuracy_cifar(model_change))
    print 'dense1:',acc
    acc =[]
    for i in range(25):
        model_change = neuron_switch(model,Layer = 'dense_2',neuron_change=np.random.choice(256,2))
        model_change.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        #print 'Mutated Test accuracy: ',accuracy_cifar(model_change)
        acc.append(accuracy_cifar(model_change))
    print 'dense2:',acc