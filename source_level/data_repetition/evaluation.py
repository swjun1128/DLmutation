from keras.models import load_model
import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype('float32').reshape(-1,28,28,1)
x_test = x_test / 255
y_test = keras.utils.to_categorical(y_test, 10)
# 怎么使用保存好的模型文件
model = keras.models.load_model("C:\\Users\\13502\DLmutation\source_level\LeNet_raw.hdf5")
score = model.evaluate(x_test, y_test)
print('Test Loss: %.4f' % score[0])
print('Test accuracy: %.4f'% score[1])


#怎么提取预测正确的测试样例