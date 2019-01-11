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
import string
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
# modelC_raw = load_model("ModelC_raw.hdf5")
# bound_data_lst_C = get_bound_data_cifar(modelC_raw,10)
# with open("bound_data_1st_modelA","w",encoding="utf-8") as boundF:
#     boundF.write(str(bound_data_lst_A))
#     boundF.close()
# with open("bound_data_1st_modelB","w",encoding="utf-8") as boundF:
#     boundF.write(str(bound_data_lst_B))
#     boundF.close()
# with open("bound_data_1st_modelC","w",encoding="utf-8") as boundF:
#     boundF.write(str(bound_data_lst_C))
#     boundF.close()


#获取原始模型A的边界值：
with open("bound_data_1st_modelA","r",encoding="utf-8") as modelAF:
    boundmodelAStr = modelAF.readline()
    modelAF.close()
boundmodelAList = boundmodelAStr.strip(string.punctuation).split(",")
for i in range(len(boundmodelAList)):
    boundmodelAList[i] = int(boundmodelAList[i])
print(boundmodelAList)
##获取原始模型B的边界值：
with open("bound_data_1st_modelB","r",encoding="utf-8") as modelBF:
    boundmodelBStr = modelBF.readline()
    modelBF.close()
boundmodelBList = boundmodelBStr.strip(string.punctuation).split(",")
for i in range(len(boundmodelBList)):
    boundmodelBList[i] = int(boundmodelBList[i])
print(boundmodelBList)
#获取原始模型C的边界值：
with open("bound_data_1st_modelC","r",encoding="utf-8") as modelCF:
    boundmodelCStr = modelCF.readline()
    modelCF.close()
boundmodelCList = boundmodelCStr.strip(string.punctuation).split(",")
for i in range(len(boundmodelCList)):
    boundmodelCList[i] = int(boundmodelCList[i])
print(boundmodelCList)
#获取变异模型：
   #获取去除激活函数的变异模型
        # 模型A
# path_1 = "Activition Function Removal\modelA\\"
# for i in range(5):
#     path_2 = "ModelA_AFR"+str(i)+".hdf5"
#     path_final = path_1+path_2
#     AFR_modelA = load_model(path_final)
#     AFR_A_acc = accuracy_in_bound_data_mnist(AFR_modelA,boundmodelAList)
#     with open(path_1+"boundedResults.txt","a+",encoding="utf-8") as bFF:
#         bFF.write(str(AFR_A_acc))
#         bFF.write("\n")
#         bFF.close()


# 模型B
# path_1 = "Activition Function Removal\modelB\\"
# for i in range(6):
#     path_2 = "ModelB_AFR"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     AFR_modelB = load_model(path_final)
#     AFR_B_acc = accuracy_in_bound_data_mnist(AFR_modelB,boundmodelBList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(AFR_B_acc))
#         bFF.write("\n")
#         bFF.close()


# 模型C
# path_1 = "Activition Function Removal\modelC\\"
# for i in range(7):
#     path_2 = "ModelC_AFR"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     AFR_modelC = load_model(path_final)
#     AFR_C_acc = accuracy_in_bound_data_cifar(AFR_modelC,boundmodelCList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(AFR_C_acc))
#         bFF.write("\n")
#         bFF.close()

path1 = ["Data Missing","Data shuffle","data_repetition","Label Error"]
path2 = ["Mutated_model","mutated_models","mutated_model","mutated models"]
path3 = ["modelA","modelB","modelC"]
path4 = ["mutated_modelA","mutated_modelB","mutated_modelC"]

# # Data Missing + ModelA
# path_1 = path1[0] +"\\"+ path2[0]+"\\" + path3[0]+"\\"
# for i in range(10):
#     path_2 = "ModelA_DM"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     DM_ModelA = load_model(path_final)
#     DM_A_acc = accuracy_in_bound_data_mnist(DM_ModelA,boundmodelAList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(DM_A_acc))
#         bFF.write("\n")
#         bFF.close()

# Data Missing + ModelB
# path_1 = path1[0] +"\\"+ path2[0]+"\\" + path3[1]+"\\"
# for i in range(10):
#     path_2 = "ModelB_DM"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     DM_ModelB = load_model(path_final)
#     DM_B_acc = accuracy_in_bound_data_mnist(DM_ModelB,boundmodelBList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(DM_B_acc))
#         bFF.write("\n")
#         bFF.close()


# Data Missing + ModelC
# path_1 = path1[0] +"\\"+ path2[0]+"\\" + path3[2]+"\\"
# for i in range(10):
#     path_2 = "ModelC_DM"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     DM_ModelC = load_model(path_final)
#     DM_C_acc = accuracy_in_bound_data_cifar(DM_ModelC,boundmodelCList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(DM_C_acc))
#         bFF.write("\n")
#         bFF.close()


# Data Shuffle + ModelA
# path_1 = path1[1] +"\\"+ path2[1]+"\\" + path3[0]+"\\"
# for i in range(10):
#     path_2 = "ModelA_RS"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     RS_ModelA = load_model(path_final)
#     RS_A_acc = accuracy_in_bound_data_mnist(RS_ModelA,boundmodelAList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(RS_A_acc))
#         bFF.write("\n")
#         bFF.close()


# Data Shuffle + ModelB
# path_1 = path1[1] +"\\"+ path2[1]+"\\" + path3[1]+"\\"
# for i in range(10):
#     path_2 = "ModelB_RS"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     RS_ModelB = load_model(path_final)
#     RS_B_acc = accuracy_in_bound_data_mnist(RS_ModelB,boundmodelBList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(RS_B_acc))
#         bFF.write("\n")
#         bFF.close()


# Data Shuffle + ModelC
# path_1 = path1[1] +"\\"+ path2[1]+"\\" + path3[2]+"\\"
# for i in range(10):
#     path_2 = "ModelC_RS"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     RS_ModelC = load_model(path_final)
#     RS_C_acc = accuracy_in_bound_data_cifar(RS_ModelC,boundmodelCList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(RS_C_acc))
#         bFF.write("\n")
#         bFF.close()


# Data_repetition + modelA
# path_1 = path1[2] +"\\"+ path2[2]+"\\" + path4[0]+"\\"
# for i in range(10):
#     path_2 = "ModelA_DR"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     DR_ModelA = load_model(path_final)
#     DR_A_acc = accuracy_in_bound_data_mnist(DR_ModelA,boundmodelAList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(DR_A_acc))
#         bFF.write("\n")
#         bFF.close()

# Data Repetition + ModelB
# path_1 = path1[2] +"\\"+ path2[2]+"\\" + path4[1]+"\\"
# for i in range(10):
#     path_2 = "ModelB_DR"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     DR_ModelB = load_model(path_final)
#     DR_B_acc = accuracy_in_bound_data_mnist(DR_ModelB,boundmodelBList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(DR_B_acc))
#         bFF.write("\n")
#         bFF.close()

# Data Repetition + ModelC
# path_1 = path1[2] +"\\"+ path2[2]+"\\" + path4[2]+"\\"
# for i in range(10):
#     path_2 = "ModelC_DR"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     DR_ModelC = load_model(path_final)
#     DR_C_acc = accuracy_in_bound_data_cifar(DR_ModelC,boundmodelCList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(DR_C_acc))
#         bFF.write("\n")
#         bFF.close()


# Label Error + modelA
# path_1 = path1[3] +"\\"+ path2[3]+"\\" + path4[0]+"\\"
# for i in range(10):
#     path_2 = "ModelA_LE"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     LE_ModelA = load_model(path_final)
#     LE_A_acc = accuracy_in_bound_data_mnist(LE_ModelA,boundmodelAList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(LE_A_acc))
#         bFF.write("\n")
#         bFF.close()


# Label Error + modelB
# path_1 = path1[3] +"\\"+ path2[3]+"\\" + path4[1]+"\\"
# for i in range(10):
#     path_2 = "ModelB_LE"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     LE_ModelB = load_model(path_final)
#     LE_B_acc = accuracy_in_bound_data_mnist(LE_ModelB,boundmodelBList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(LE_B_acc))
#         bFF.write("\n")
#         bFF.close()


# Label Error + model C
# path_1 = path1[3] +"\\"+ path2[3]+"\\" + path4[2]+"\\"
# for i in range(10):
#     path_2 = "ModelC_LE"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     LE_ModelC = load_model(path_final)
#     LE_C_acc = accuracy_in_bound_data_cifar(LE_ModelC,boundmodelCList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(LE_C_acc))
#         bFF.write("\n")
#         bFF.close()


# Layer Add + modelA
# path_1 = "Layer Add\modelA\\"
# for i in range(10):
#     path_2 = "ModelA_LA"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     LA_modelA = load_model(path_final)
#     LA_A_acc = accuracy_in_bound_data_mnist(LA_modelA,boundmodelAList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(LA_A_acc))
#         bFF.write("\n")
#         bFF.close()



# Layer Add + modelB
# path_1 = "Layer Add\modelB\\"
# for i in range(10):
#     path_2 = "ModelB_LA"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     LA_modelB = load_model(path_final)
#     LA_B_acc = accuracy_in_bound_data_mnist(LA_modelB,boundmodelBList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(LA_B_acc))
#         bFF.write("\n")
#         bFF.close()



# Layer Add + modelC
# path_1 = "Layer Add\modelC\\"
# for i in range(20):
#     path_2 = "ModelC_LA"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     LA_modelC = load_model(path_final)
#     LA_C_acc = accuracy_in_bound_data_cifar(LA_modelC,boundmodelCList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(LA_C_acc))
#         bFF.write("\n")
#         bFF.close()


# # Layer Removal + modelA
# path_1 = "Layer Removal\ModelA\\"
# for i in range(10):
#     path_2 = "ModelA_LR"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     LR_modelA = load_model(path_final)
#     LR_A_acc = accuracy_in_bound_data_mnist(LR_modelA,boundmodelAList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(LR_A_acc))
#         bFF.write("\n")
#         bFF.close()

# Layer Removal + modelB
# path_1 = "Layer Removal\ModelB\\"
# for i in range(10):
#     path_2 = "ModelB_LR"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     LR_modelB = load_model(path_final)
#     LR_B_acc = accuracy_in_bound_data_mnist(LR_modelB,boundmodelBList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(LR_B_acc))
#         bFF.write("\n")
#         bFF.close()

# Layer Removal + modelC
# path_1 = "Layer Removal\ModelC\\"
# for i in range(10):
#     path_2 = "ModelC_LR"+str(i)+".hdf5"
#     path_final = path_1 + path_2
#     LR_modelC = load_model(path_final)
#     LR_C_acc = accuracy_in_bound_data_cifar(LR_modelC,boundmodelCList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(LR_C_acc))
#         bFF.write("\n")
#         bFF.close()


# Noise Perturbation + modelA
# path_1 = "Noise Perturbation\modelA\\"
# for i in range(10):
#     path_2 = "ModelA_NP" + str(i) + ".hdf5"
#     path_final = path_1 + path_2
#     NP_modelA = load_model(path_final)
#     NP_A_acc = accuracy_in_bound_data_mnist(NP_modelA,boundmodelAList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(NP_A_acc))
#         bFF.write("\n")
#         bFF.close()


# Noise Perturbation + modelB
# path_1 = "Noise Perturbation\modelB\\"
# for i in range(10):
#     path_2 = "ModelB_NP" + str(i) + ".hdf5"
#     path_final = path_1 + path_2
#     NP_modelB = load_model(path_final)
#     NP_B_acc = accuracy_in_bound_data_mnist(NP_modelB,boundmodelBList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(NP_B_acc))
#         bFF.write("\n")
#         bFF.close()

# Noise Perturbation + modelC
# path_1 = "Noise Perturbation\modelC\\"
# for i in range(10):
#     path_2 = "ModelC_NP" + str(i) + ".hdf5"
#     path_final = path_1 + path_2
#     NP_modelC = load_model(path_final)
#     NP_C_acc = accuracy_in_bound_data_cifar(NP_modelC,boundmodelCList)
#     with open(path_1 + "boundedResults.txt", "a+", encoding="utf-8") as bFF:
#         bFF.write(str(NP_C_acc))
#         bFF.write("\n")
#         bFF.close()