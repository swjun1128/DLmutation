#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.models import Model,Input,load_model
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from tqdm import tqdm
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

test=mnist.test.images

def cover(model,test,threshold=0.2,choice_index=[]):
    '''
    model_path:模型路径
    test：测试集合
    threhold：选择的阀值
    example：随机选择训练集合的数量
    '''
    lst=[1,3,5]
    #print model.layers
    layer_num=0
    for index in lst:
        layer_num+=int(model.layers[index].output.shape[-1])

    model_layer=Model(inputs=model.input,outputs=[model.layers[i].output for i in lst])
    image=test[choice_index]
    act_layers=model_layer.predict_on_batch(image)
    #print type(act_layers)
    #threshold=0.99999999999999999
    act_num=0
    max_output=np.zeros(1000)
    print type(max_output)
    for act in act_layers:
        max_output=np.maximum(max_output,np.max(act,axis=1))
    
    for act in act_layers:
        print 'threshold:',threshold
        act_num+=(np.sum((act.T>max_output*threshold).T,axis=0)>0).sum()
    print act_num
    ratio=act_num/float(layer_num)
    return ratio


def find_second_index(act):
    max_=0
    second_max=0
    index=0
    for i in range(10):
        if act[i]>max_:
            max_=act[i]
        elif act[i]>second_max:#else表示不是最大的那个index
            second_max=act[i]
            index=i
    ratio=1.0*second_max/max_
    return index,ratio#ratio是第二大输出达到最大输出的百分比

def second_cover(model,test,ratio_threshold=0,choice_index=[]):
    '''
    model_path:模型路径
    test：测试集合
    threhold：选择的阀值
    example：随机选择训练集合的数量
    '''
    iscnn=False
    out_index=len(model.layers)-1
    model.layers[out_index]
    model_layer=Model(inputs=model.input,outputs=model.layers[out_index].output)
    image=test[choice_index]
    if iscnn==True:
        image=image.reshape(-1,28,28,1)
    act_layers=model_layer.predict_on_batch(image)
    pred=model.predict(image)
    pred=list(map(lambda x:np.argmax(x),pred))
    label=list(map(lambda x:np.argmax(x),mnist.test.labels[choice_index]))
    dic={}
    for i in range(10):
        dic[i]=set()
    for i in range(len(act_layers)):#此i只是choice_index序化后
        act=act_layers[i]

        second_index,ratio = find_second_index(act)
        if ratio>ratio_threshold:
            dic[label[i]].add(second_index)
        
    cover_num=0
    for i in range(10):
        cover_num+=len(dic[i])
    adequacy_score=1.0*cover_num/90#每一个标签对应都有9个第二大的index
    return adequacy_score


def accuracy(model,test,choice_index):
    pred=model.predict(test)
    pred=list(map(lambda x:np.argmax(x),pred))
    test_label=list(map(lambda x:np.argmax(x),mnist.test.labels))
    acc=0
    for i in choice_index:
        if pred[i]==test_label[i]:
            acc+=1
    return 1.0*acc/len(choice_index)

if __name__=='__main__':
    #ratio=cover('model/model.hdf5',test)
    #print ratio
    model_path='./model/model.hdf5'
    model=load_model(model_path)
    ratio_threshold=0.9
    #choice_index = np.random.choice(range(10000),size=500,replace=False)
    choice_index = np.random.choice(range(10000),size=1000,replace=False)
    #choice_index=[i for i in range(10000)]
    score= second_cover(model,test,ratio_threshold,choice_index)
    print score
