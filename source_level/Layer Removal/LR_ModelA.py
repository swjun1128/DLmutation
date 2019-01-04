# -*- coding: utf-8 -*-
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split

def getModel(k):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

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
    #model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()


    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1,shuffle=False)  #shuffle=True,
    score = model.evaluate(x_test, y_test)
    print('Test Loss: %.4f' % score[0])
    print('Test accuracy: %.4f'% score[1])
    f = open("ModelA_LR_84.txt", "a+")
    f.write('Test Loss: %.4f' % score[0])
    f.write("\t")
    f.write('Test accuracy: %.4f' % score[1])
    f.write("\n")
    f.close()
    model.save('ModelA_LR'+str(k)+'.hdf5')

for i in range(5,10):
    getModel(i)
