import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import random


def getModel(k):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 记录随机抽取的Index
    randomIndex = []
    for i in range(600):
        randomIndex.append(random.randint(0,59999))
    print(randomIndex)
    f = open("ModelB_randomIndex_Record.txt","a+")
    for i in randomIndex:
        f.write(str(i)+"   ")
    f.write("\n")

    tempArray_x = np.zeros((600,28,28))
    tempArray_y = np.zeros((600,1))

    for i in range(600):
        tempArray_x[i] = x_train[randomIndex[i]]
        tempArray_y[i] = y_train[randomIndex[i]]
    indices =np.random.permutation(tempArray_x.shape[0])
    rand_data_x = tempArray_x[indices]
    rand_data_y = tempArray_y[indices]

    for i in range(600):
        if (x_train[randomIndex[i]] != rand_data_x[i]).any():
            print("Different Success!!!!")
        x_train[randomIndex[i]] = rand_data_x[i]
        y_train[randomIndex[i]] = rand_data_y[i]

    # 输入数据为 mnist 数据集
    x_train = x_train.astype('float32').reshape(-1,28,28,1)
    x_test = x_test.astype('float32').reshape(-1,28,28,1)


    x_train = x_train / 255
    x_test = x_test / 255


    print('Train:{},Test:{}'.format(len(x_train),len(x_test)))
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)


    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',padding='valid', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),padding='valid', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1,shuffle=False)
    score = model.evaluate(x_test, y_test)
    print('Test Loss: %.4f' % score[0])
    print('Test accuracy: %.4f'% score[1])
    f.write('Test Loss: %.4f' % score[0])
    f.write("\t")
    f.write('Test accuracy: %.4f'% score[1])
    f.write("\n")
    f.close()
    model.save("ModelB_RS"+str(k)+".hdf5")


for i in range(10):
    getModel(i)



