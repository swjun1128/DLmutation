import keras
from keras.datasets import mnist
import numpy as np
import  random
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential

#MNIST共有60000训练数据，10000测试数据

# seeds = []
# for i in range(10):
#     seeds.append(random.randint(1,1000))
# print(seeds)
# [432, 919, 57, 757, 215, 738, 430, 357, 326, 55]




# 分离出训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # y_train 的shape：(60000,)
#定义随即复制变量
x_train_re = np.zeros((6000,28,28))
y_train_re = np.zeros((6000,))
#制定随机种子（就是上面的seedslist中的值），便于复现
random.seed(432)    #
ranIndexList = []
for i in range(6000):
    temp = random.randint(0,59999)
    ranIndexList.append(temp)
    x_train_re[i] = x_train[temp]
    y_train_re[i] = y_train[temp]
    # print(x_train_re.shape)  (28, 28)
print(ranIndexList)   #打印不同随机种子生成的随机采样结果。
x_train = np.concatenate((x_train_re, x_train), axis=0)
y_train = np.concatenate((y_train_re, y_train), axis=0)


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
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1) #shuffle=True,
score = model.evaluate(x_test, y_test)
print('Test Loss: %.4f' % score[0])
print('Test accuracy: %.4f'% score[1])
model.save('ModelB_Seed432.hdf5')