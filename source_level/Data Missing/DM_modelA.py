import keras
import random
from keras.datasets import mnist
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# seeds = [] [415, 87, 622, 307, 509, 224, 904, 9, 598, 825]
DataMiss_Index = []
random.seed(825)
DataMiss_Index = random.sample(range(60000),600)
x_train = np.delete(x_train,DataMiss_Index,axis=0)
y_train = np.delete(y_train,DataMiss_Index,axis=0)

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
model.save('ModelA_DM9.hdf5')












