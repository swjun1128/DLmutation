#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:42:34 2018

@author: qq
"""

import h5py  #导入工具包  
import numpy as np  
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