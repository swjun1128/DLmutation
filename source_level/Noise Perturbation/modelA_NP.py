# -*- coding: utf-8 -*-
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import random
import numpy as np

def getModel(k):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #噪声
    #随机抽取1%数据 600
    randomIndex = []
    for i in range(600):
        randomIndex.append(random.randint(0,59999))
    print(randomIndex)
    #噪声扰动
    with open("modelA\A"+str(k)+"\\"+"randomResult_"+str(k)+".txt","a+",encoding="utf-8") as f:
        for i in range(600):
            f.write(str(randomIndex[i]))
            f.write("\t")
        f.write("\n")
        f.close()

    #中间结果
    tempResult = np.zeros((600,28,28))
    for i in range(600):
        tempResult[i] = x_train[randomIndex[i]]
    #写入文件中
    with open("modelA\A"+str(k)+"\\"+"np_"+str(k)+".txt","a+",encoding="utf-8") as f:
        for i in range(600):
            np.savetxt(f,x_train[randomIndex[i]])
            f.write("\n")
            f.write("\n")
            f.write("\n")
        f.close()

    #噪声扰动
    #就是随机取一个中心点，实现9*9的随机取值(60000*28*28)

    tempCenterSet = []
    for i in range(600):
        tempCenter = []
        x_axis = random.randint(1,26)
        y_axis = random.randint(1,26)
        while x_train[randomIndex[i]][x_axis][y_axis] == 0:
            print("重新随机")
            x_axis = random.randint(1, 26)
            y_axis = random.randint(1, 26)
        print(x_axis)
        print(y_axis)
        print(x_train[randomIndex[i]][int(x_axis) ][int(y_axis) ])
        print("\n")
        tempCenter.append(x_axis)
        tempCenter.append(y_axis)
        tempCenterSet.append(tempCenter)
    with open("modelA\A"+str(k)+"\\"+"randomResult_"+str(k)+".txt","a+",encoding="utf-8") as f :
        for i in range(600):
            f.write(str(tempCenterSet[i]))
            f.write("\t")
        f.write("\t")
        f.write("\n")
        f.close()

    #处理3*3的随机取值
    #用numpy.random 随机生成3*3的255之间的数组
    for i in range(600):
        npArray = np.random.randint(0, 255, (3, 3), np.uint8)
        with open("modelA\A"+str(k)+"\\"+"npArray_"+str(k)+".txt","a+",encoding="utf-8") as arrayF:
            np.savetxt(arrayF,npArray)
            arrayF.write("\n")
            arrayF.close()
        tempIndex = randomIndex[i]
        tempX = tempCenterSet[i][0]
        tempY = tempCenterSet[i][1]
        #为了验证随机扰动成功，存储原有的值
        initialArray = np.zeros((3,3),np.uint8)
        initialArray[0][0] = x_train[tempIndex][int(tempX) - 1][int(tempY) - 1]
        initialArray[0][1] = x_train[tempIndex][int(tempX) - 1][int(tempY) ]
        initialArray[0][2] = x_train[tempIndex][int(tempX) - 1][int(tempY) + 1]
        initialArray[1][0] = x_train[tempIndex][int(tempX) ][int(tempY) - 1]
        initialArray[1][1] = x_train[tempIndex][int(tempX) ][int(tempY) ]
        initialArray[1][2] = x_train[tempIndex][int(tempX) ][int(tempY) + 1]
        initialArray[2][0] = x_train[tempIndex][int(tempX) +1][int(tempY) - 1]
        initialArray[2][1] = x_train[tempIndex][int(tempX) +1][int(tempY) ]
        initialArray[2][2] = x_train[tempIndex][int(tempX) +1][int(tempY) +1]
        with open("modelA\A"+str(k)+"\\"+"initialArray_"+str(k)+".txt","a+",encoding="utf-8") as initialF:
            np.savetxt(initialF,initialArray)
            initialF.write("\n")
            initialF.write("\n")
            initialF.write("\n")
            initialF.close()
        # 变异--噪声扰动
        x_train[tempIndex][int(tempX) - 1][int(tempY) - 1] = npArray[0][0]
        x_train[tempIndex][int(tempX) - 1][int(tempY) ] = npArray[0][1]
        x_train[tempIndex][int(tempX) - 1][int(tempY) + 1] = npArray[0][2]
        x_train[tempIndex][int(tempX) ][int(tempY) - 1] = npArray[1][0]
        x_train[tempIndex][int(tempX) ][int(tempY) ] = npArray[1][1]
        x_train[tempIndex][int(tempX) ][int(tempY) + 1] = npArray[1][2]
        x_train[tempIndex][int(tempX) +1][int(tempY) - 1] = npArray[2][0]
        x_train[tempIndex][int(tempX) +1][int(tempY) ] = npArray[2][1]
        x_train[tempIndex][int(tempX) +1][int(tempY) +1] = npArray[2][2]

        #存储中间结果至npedArray.txt
        npedArray = np.zeros((3,3),np.uint8)
        npedArray[0][0] = x_train[tempIndex][int(tempX) - 1][int(tempY) - 1]
        npedArray[0][1] = x_train[tempIndex][int(tempX) - 1][int(tempY)]
        npedArray[0][2] = x_train[tempIndex][int(tempX) - 1][int(tempY) + 1]
        npedArray[1][0] = x_train[tempIndex][int(tempX)][int(tempY) - 1]
        npedArray[1][1] = x_train[tempIndex][int(tempX)][int(tempY)]
        npedArray[1][2] = x_train[tempIndex][int(tempX)][int(tempY) + 1]
        npedArray[2][0] = x_train[tempIndex][int(tempX) + 1][int(tempY) - 1]
        npedArray[2][1] = x_train[tempIndex][int(tempX) + 1][int(tempY)]
        npedArray[2][2] = x_train[tempIndex][int(tempX) + 1][int(tempY) + 1]
        with open("modelA\A"+str(k)+"\\"+"npedArray_"+str(k)+".txt", "a+", encoding="utf-8") as npedF:
            np.savetxt(npedF, npedArray)
            npedF.write("\n")
            npedF.write("\n")
            npedF.write("\n")
            npedF.close()


    # 输入数据为 mnist 数据集
    x_train = x_train.astype('float32').reshape(-1,28,28,1)
    x_test = x_test.astype('float32').reshape(-1,28,28,1)

    x_train = x_train / 255
    x_test = x_test / 255
    print('Train:{},Test:{}'.format(len(x_train),len(x_test)))
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu',padding='valid',input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5),padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()


    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1,shuffle=False)  #shuffle=True,
    score = model.evaluate(x_test, y_test)
    print('Test Loss: %.4f' % score[0])
    print('Test accuracy: %.4f'% score[1])
    with open('modelA\\'+'ModelAResults.txt',"a+",encoding="utf-8") as result_f :
        result_f.write('Test Loss: %.4f' % score[0])
        result_f.write("\t")
        result_f.write('Test accuracy: %.4f'% score[1])
        result_f.write("\n")
        result_f.close()
    model.save('modelA\\'+'ModelA_NP'+str(k)+'.hdf5')

for i in range(10):
    getModel(i)