#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.models import Model,Input,load_model
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
from tqdm import tqdm
import foolbox
import time


mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

testimages=mnist.test.images

#这个文档要做的事情是研究对抗样本生成的难度是否和原始样本的优劣有关
#原始样本的第二大输出越接近第一，生成对抗样本的难度越小，表现为生成对抗样本的算法运算时间越短


def find_second_ratio(model,image):
    iscnn=False
    out_index=len(model.layers)-1
    model.layers[out_index]
    model_layer=Model(inputs=model.input,outputs=model.layers[out_index].output)
    if iscnn==True:
        image=image.reshape(-1,28,28,1)
    act_layers=model_layer.predict_on_batch(image)
    act=act_layers[0] 
    max_=0
    second_max=0
    sec_index=0
    for i in range(10):
        if act[i]>max_:
            max_=act[i]
            max_index=i
        elif act[i]>second_max:#else表示不是最大的那个index
            second_max=act[i]
            sec_index=i
    ratio=1.0*second_max/max_
    #return max_,second_max,max_index,sec_index,ratio#ratio是第二大输出达到最大输出的百分比
    return max_index,sec_index,ratio

def generate_adv(model,index):#label是预测值，
    #label=np.argmax(model.predict(np.expand_dims(img,axis=0)))
    label=np.argmax(mnist.test.labels[index])
    foolmodel=foolbox.models.KerasModel(model,bounds=(0,1),preprocessing=(0,1))
    attack=foolbox.attacks.IterativeGradientAttack(foolmodel)
    img=testimages[index]
    adv=attack(img,label,epsilons=[0.01,0.1,1],steps=100)
    return adv


                
if __name__=='__main__':
    model_path='./model/model_raw.hdf5'
    model=load_model(model_path)
    label=np.argmax(mnist.test.labels,axis=1)
    pred=np.argmax(model.predict(testimages),axis=1)
    goodcase=pred==label
    goodindex=[]
    diff_time=[]
    #adv_lst=[]
    adv_label_lst=[]
    right_label_lst=[]
    sec_index_label=[]
    ratio_lst=[]
    for i in range(10000):
        if goodcase[i]==True:
            goodindex.append(i)
    for i in goodindex:
        print i
        if i <8500:
            continue
        before =time.time()
        adv_img=generate_adv(model,i)
        after =time.time()
        image=testimages[[i]]
        max_index,sec_index,ratio = find_second_ratio(model,image)
        ratio_lst.append(ratio)
        if isinstance(adv_img,np.ndarray):
            adv_label=np.argmax(model.predict(np.expand_dims(adv_img,axis=0)))
            adv_label_lst.append(adv_label)
            right_label_lst.append(max_index)
            sec_index_label.append(sec_index)
            diff =after-before
            diff_time.append(diff)   
    result=[]
    for i in range(len(adv_label_lst)):
        result.append([right_label_lst[i],sec_index_label[i],ratio_lst[i],adv_label_lst[i],diff_time[i]])
    csv = pd.DataFrame(result,columns=['origin','second','ratio','advlabel','time'])      
    csv.to_csv('adv_coverage8.csv')
