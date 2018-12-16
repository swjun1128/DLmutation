# DLmutation
Deep learning mutation empirical study


目前对照DeepMutation的三种模型结构，LeNet5.py 、Model_B.py、 ModelC.py(目前精度是74左右)    
关于训练的问题，验证集怎么选。---scikit-learn来将训练集选出1%作为验证集，不参加训练。  (我等会试一下不用验证集，之前的是使用测试集作为验证集，感觉有点问题)  
训练集：  54000  
验证集：  6000  
测试集:   10000  
1. 数据复制： 随机复制1% Mutation_Re.py
