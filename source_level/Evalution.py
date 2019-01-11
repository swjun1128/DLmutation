from keras.datasets import mnist
import keras
import random
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import np_utils
from boundary import get_bound_data_mnist
from boundary import accuracy_in_bound_data_mnist
from boundary import get_bound_data_cifar
from boundary import accuracy_in_bound_data_cifar
def getEvaMinst(path1,path2):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32').reshape(-1,28,28,1)
    x_test = x_test.astype('float32').reshape(-1,28,28,1)
    x_train = x_train / 255
    x_test = x_test / 255
    print('Train:{},Test:{}'.format(len(x_train),len(x_test)))
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    model_path = path1+path2
    model = load_model(model_path)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('测试集 score(val_loss): %.4f' % score[0])
    print('测试集 accuracy: %.4f' % score[1])
    with open(path1+"results.txt","a+",encoding="utf-8") as rf:
        rf.write('测试集 score(val_loss): %.4f' % score[0])
        rf.write("\t")
        rf.write('测试集 accuracy: %.4f' % score[1])
        rf.write("\n")
        rf.close()

def getCifarEva(path1,path2):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32×32
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('Train:{},Test:{}'.format(len(X_train), len(X_test)))
    nb_classes = 10
    # convert integers to dummy variables (one hot encoding)
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    print('data success')
    model_path = path1 + path2
    model = load_model(model_path)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('测试集 score(val_loss): %.4f' % score[0])
    print('测试集 accuracy: %.4f' % score[1])
    with open(path1 + "results.txt", "a+", encoding="utf-8") as rf:
        rf.write('测试集 score(val_loss): %.4f' % score[0])
        rf.write("\t")
        rf.write('测试集 accuracy: %.4f' % score[1])
        rf.write("\n")
        rf.close()

# path1 = ["Data Missing","Data shuffle","data_repetition","Label Error"]
# path2 = ["Mutated_model","mutated_models","mutated_model","mutated models"]
# path3 = ["modelA","modelB","modelC"]
# path4 = ["mutated_modelA","mutated_modelB","mutated_modelC"]

# for i in range(10):
#     path5 = ".\\"+path1[3]+"\\"+path2[3]+"\\"+path4[0]+"\\"
#     path6 = "ModelA_LE"+str(i)+".hdf5"
#     getEvaMinst(path5,path6)
# for i in range(10):
#     path5 = ".\\"+path1[3]+"\\"+path2[3]+"\\"+path4[1]+"\\"
#     path6 = "ModelB_LE"+str(i)+".hdf5"
#     getEvaMinst(path5,path6)
# for i in range(10):
#     path5 = ".\\"+path1[3]+"\\"+path2[3]+"\\"+path4[2]+"\\"
#     path6 = "ModelC_LE"+str(i)+".hdf5"
#     getCifarEva(path5,path6)



#边界值度量
# modelA_raw = load_model("ModelA_raw.hdf5")
# bound_data_lst_A = get_bound_data_mnist(modelA_raw,10)
# modelB_raw = load_model("ModelB_raw.hdf5")
# bound_data_lst_B = get_bound_data_mnist(modelB_raw,10)
modelC_raw = load_model("ModelC_raw.hdf5")
bound_data_lst_C = get_bound_data_cifar(modelC_raw,10)
# with open("bound_data_1st_modelA","w",encoding="utf-8") as boundF:
#     boundF.write(str(bound_data_lst_A))
#     boundF.close()
# with open("bound_data_1st_modelB","w",encoding="utf-8") as boundF:
#     boundF.write(str(bound_data_lst_B))
#     boundF.close()
# with open("bound_data_1st_modelC","w",encoding="utf-8") as boundF:
#     boundF.write(str(bound_data_lst_C))
#     boundF.close()

#get 去除激活函数后的模型A在边界值上的精确度
# for i in range(5):
#     path_AFR = "Activition Function Removal\modelA\\"
#     path_final = path_AFR + "ModelA_AFR"+str(i)+".hdf5"
#     model_mutated = load_model(path_final)
#     acc = accuracy_in_bound_data_mnist(model_mutated,bound_data_lst)
#     with open(path_AFR+"bound_results.txt","a+",encoding="utf-8") as boundResF:
#         boundResF.write(str(acc))
#         boundResF.write("\n")
#         boundResF.close()


#get 去除激活函数后的模型B在边界值上的精确度
# for i in range(6):
#     path_AFR = "Activition Function Removal\modelB\\"
#     path_final = path_AFR + "ModelB_AFR"+str(i)+".hdf5"
#     model_mutated = load_model(path_final)
#     acc = accuracy_in_bound_data_mnist(model_mutated,bound_data_lst)
#     with open(path_AFR+"bound_results.txt","a+",encoding="utf-8") as boundResF:
#         boundResF.write(str(acc))
#         boundResF.write("\n")
#         boundResF.close()

#get 去除激活函数后的模型C在边界值上的精确度
# for i in range(7):
#     path_AFR = "Activition Function Removal\modelC\\"
#     path_final = path_AFR + "ModelC_AFR"+str(i)+".hdf5"
#     model_mutated = load_model(path_final)
#     acc = accuracy_in_bound_data_cifar(model_mutated,bound_data_lst)
#     with open(path_AFR+"bound_results.txt","a+",encoding="utf-8") as boundResF:
#         boundResF.write(str(acc))
#         boundResF.write("\n")
#         boundResF.close()

#get 删除数据后的模型A.B.C在边界值上的精确度

# path1 = ["Data Missing","Data shuffle","data_repetition","Label Error"]
# path2 = ["Mutated_model","mutated_models","mutated_model","mutated models"]
# path3 = ["modelA","modelB","modelC"]
# path4 = ["mutated_modelA","mutated_modelB","mutated_modelC"]
#
# for i in range(10):
#     path5 = path1[0]+"\\"+path2[0]+"\\"+path3[0]+"\\"
#     path6 = "ModelA_DM"+str(i)+".hdf5"
#     model_DM = load_model(path5+path6)
#     acc = accuracy_in_bound_data_cifar(model_mutated, bound_data_lst)
#     with open(path_AFR+"bound_results.txt","a+",encoding="utf-8") as boundResF:
#         boundResF.write(str(acc))
#         boundResF.write("\n")
#         boundResF.close()
#
# for i in range(10):
#     path5 = ".\\"+path1[3]+"\\"+path2[3]+"\\"+path4[1]+"\\"
#     path6 = "ModelB_LE"+str(i)+".hdf5"
#     getEvaMinst(path5,path6)
# for i in range(10):
#     path5 = ".\\"+path1[3]+"\\"+path2[3]+"\\"+path4[2]+"\\"
#     path6 = "ModelC_LE"+str(i)+".hdf5"
#     getCifarEva(path5,path6)