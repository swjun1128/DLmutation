from keras.datasets import mnist
from keras.datasets import cifar10
import keras
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
DR_modelA =[]
DR_modelB =[]
DR_modelC =[]
LE_modelA =[]
LE_modelB =[]
LE_modelC =[]
DM_modelA =[]
DM_modelB =[]
DM_modelC =[]

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32').reshape(-1,28,28,1)
x_test = x_test.astype('float32').reshape(-1,28,28,1)
x_train = x_train / 255
x_test = x_test / 255
print('Train:{},Test:{}'.format(len(x_train),len(x_test)))
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('Train:{},Test:{}'.format(len(X_train),len(X_test)))
nb_classes=10
# convert integers to dummy variables (one hot encoding)
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
print('data success')

#
# #DR on model A
# DR_modelA_f0 = ".\source_level\data_repetition\mutated_model\mutated_modelA\LeNet_Rep_Seed55.hdf5"
# DR_modelA_f1 = ".\source_level\data_repetition\mutated_model\mutated_modelA\LeNet_Rep_Seed57.hdf5"
# DR_modelA_f2 = ".\source_level\data_repetition\mutated_model\mutated_modelA\LeNet_Rep_Seed215.hdf5"
# DR_modelA_f3 = ".\source_level\data_repetition\mutated_model\mutated_modelA\LeNet_Rep_Seed326.hdf5"
# DR_modelA_f4 = ".\source_level\data_repetition\mutated_model\mutated_modelA\LeNet_Rep_Seed357.hdf5"
# DR_modelA_f5 = ".\source_level\data_repetition\mutated_model\mutated_modelA\LeNet_Rep_Seed430.hdf5"
# DR_modelA_f6 = ".\source_level\data_repetition\mutated_model\mutated_modelA\LeNet_Rep_Seed432.hdf5"
# DR_modelA_f7 = ".\source_level\data_repetition\mutated_model\mutated_modelA\LeNet_Rep_Seed738.hdf5"
# DR_modelA_f8 = ".\source_level\data_repetition\mutated_model\mutated_modelA\LeNet_Rep_Seed757.hdf5"
# DR_modelA_f9 = ".\source_level\data_repetition\mutated_model\mutated_modelA\LeNet_Rep_Seed919.hdf5"
#
#
# model0 = load_model(DR_modelA_f0)
# model1 = load_model(DR_modelA_f1)
# model2 = load_model(DR_modelA_f2)
# model3 = load_model(DR_modelA_f3)
# model4 = load_model(DR_modelA_f4)
# model5 = load_model(DR_modelA_f5)
# model6 = load_model(DR_modelA_f6)
# model7 = load_model(DR_modelA_f7)
# model8 = load_model(DR_modelA_f8)
# model9 = load_model(DR_modelA_f9)
# score0 = model0.evaluate(x_test, y_test, verbose=0)
# score1 = model1.evaluate(x_test, y_test, verbose=0)
# score2 = model2.evaluate(x_test, y_test, verbose=0)
# score3 = model3.evaluate(x_test, y_test, verbose=0)
# score4 = model4.evaluate(x_test, y_test, verbose=0)
# score5 = model5.evaluate(x_test, y_test, verbose=0)
# score6 = model6.evaluate(x_test, y_test, verbose=0)
# score7 = model7.evaluate(x_test, y_test, verbose=0)
# score8 = model8.evaluate(x_test, y_test, verbose=0)
# score9 = model9.evaluate(x_test, y_test, verbose=0)
# DR_modelA.append(score0)
# DR_modelA.append(score1)
# DR_modelA.append(score2)
# DR_modelA.append(score3)
# DR_modelA.append(score4)
# DR_modelA.append(score5)
# DR_modelA.append(score6)
# DR_modelA.append(score7)
# DR_modelA.append(score8)
# DR_modelA.append(score9)
# print(DR_modelA)
#
# # DR on model B
# DR_modelB_f0 = ".\source_level\data_repetition\mutated_model\mutated_modelB\ModelB_Seed55.hdf5"
# DR_modelB_f1 = ".\source_level\data_repetition\mutated_model\mutated_modelB\ModelB_Seed57.hdf5"
# DR_modelB_f2 = ".\source_level\data_repetition\mutated_model\mutated_modelB\ModelB_Seed215.hdf5"
# DR_modelB_f3 = ".\source_level\data_repetition\mutated_model\mutated_modelB\ModelB_Seed326.hdf5"
# DR_modelB_f4 = ".\source_level\data_repetition\mutated_model\mutated_modelB\ModelB_Seed357.hdf5"
# DR_modelB_f5 = ".\source_level\data_repetition\mutated_model\mutated_modelB\ModelB_Seed430.hdf5"
# DR_modelB_f6 = ".\source_level\data_repetition\mutated_model\mutated_modelB\ModelB_Seed432.hdf5"
# DR_modelB_f7 = ".\source_level\data_repetition\mutated_model\mutated_modelB\ModelB_Seed738.hdf5"
# DR_modelB_f8 = ".\source_level\data_repetition\mutated_model\mutated_modelB\ModelB_Seed757.hdf5"
# DR_modelB_f9 = ".\source_level\data_repetition\mutated_model\mutated_modelB\ModelB_Seed919.hdf5"
# model0 = load_model(DR_modelB_f0)
# model1 = load_model(DR_modelB_f1)
# model2 = load_model(DR_modelB_f2)
# model3 = load_model(DR_modelB_f3)
# model4 = load_model(DR_modelB_f4)
# model5 = load_model(DR_modelB_f5)
# model6 = load_model(DR_modelB_f6)
# model7 = load_model(DR_modelB_f7)
# model8 = load_model(DR_modelB_f8)
# model9 = load_model(DR_modelB_f9)
# score0 = model0.evaluate(x_test, y_test, verbose=0)
# score1 = model1.evaluate(x_test, y_test, verbose=0)
# score2 = model2.evaluate(x_test, y_test, verbose=0)
# score3 = model3.evaluate(x_test, y_test, verbose=0)
# score4 = model4.evaluate(x_test, y_test, verbose=0)
# score5 = model5.evaluate(x_test, y_test, verbose=0)
# score6 = model6.evaluate(x_test, y_test, verbose=0)
# score7 = model7.evaluate(x_test, y_test, verbose=0)
# score8 = model8.evaluate(x_test, y_test, verbose=0)
# score9 = model9.evaluate(x_test, y_test, verbose=0)
# DR_modelB.append(score0)
# DR_modelB.append(score1)
# DR_modelB.append(score2)
# DR_modelB.append(score3)
# DR_modelB.append(score4)
# DR_modelB.append(score5)
# DR_modelB.append(score6)
# DR_modelB.append(score7)
# DR_modelB.append(score8)
# DR_modelB.append(score9)
# print(DR_modelB)
#
#
# # DR on model C
# DR_modelC_f0 = ".\source_level\data_repetition\mutated_model\mutated_modelC\ModelC_Seed55.hdf5"
# DR_modelC_f1 = ".\source_level\data_repetition\mutated_model\mutated_modelC\ModelC_Seed57.hdf5"
# DR_modelC_f2 = ".\source_level\data_repetition\mutated_model\mutated_modelC\ModelC_Seed215.hdf5"
# DR_modelC_f3 = ".\source_level\data_repetition\mutated_model\mutated_modelC\ModelC_Seed326.hdf5"
# DR_modelC_f4 = ".\source_level\data_repetition\mutated_model\mutated_modelC\ModelC_Seed357.hdf5"
# DR_modelC_f5 = ".\source_level\data_repetition\mutated_model\mutated_modelC\ModelC_Seed430.hdf5"
# DR_modelC_f6 = ".\source_level\data_repetition\mutated_model\mutated_modelC\ModelC_Seed432.hdf5"
# DR_modelC_f7 = ".\source_level\data_repetition\mutated_model\mutated_modelC\ModelC_Seed738.hdf5"
# DR_modelC_f8 = ".\source_level\data_repetition\mutated_model\mutated_modelC\ModelC_Seed757.hdf5"
# DR_modelC_f9 = ".\source_level\data_repetition\mutated_model\mutated_modelC\ModelC_Seed919.hdf5"
# model0 = load_model(DR_modelC_f0)
# model1 = load_model(DR_modelC_f1)
# model2 = load_model(DR_modelC_f2)
# model3 = load_model(DR_modelC_f3)
# model4 = load_model(DR_modelC_f4)
# model5 = load_model(DR_modelC_f5)
# model6 = load_model(DR_modelC_f6)
# model7 = load_model(DR_modelC_f7)
# model8 = load_model(DR_modelC_f8)
# model9 = load_model(DR_modelC_f9)
# score0 = model0.evaluate(X_test, Y_test, verbose=0)
# score1 = model1.evaluate(X_test, Y_test, verbose=0)
# score2 = model2.evaluate(X_test, Y_test, verbose=0)
# score3 = model3.evaluate(X_test, Y_test, verbose=0)
# score4 = model4.evaluate(X_test, Y_test, verbose=0)
# score5 = model5.evaluate(X_test, Y_test, verbose=0)
# score6 = model6.evaluate(X_test, Y_test, verbose=0)
# score7 = model7.evaluate(X_test, Y_test, verbose=0)
# score8 = model8.evaluate(X_test, Y_test, verbose=0)
# score9 = model9.evaluate(X_test, Y_test, verbose=0)
# DR_modelC.append(score0)
# DR_modelC.append(score1)
# DR_modelC.append(score2)
# DR_modelC.append(score3)
# DR_modelC.append(score4)
# DR_modelC.append(score5)
# DR_modelC.append(score6)
# DR_modelC.append(score7)
# DR_modelC.append(score8)
# DR_modelC.append(score9)
# print(DR_modelC)

# DR_modelA =[[0.07848312225304543, 0.975], [0.07546221836628392, 0.977], [0.08415429895594716, 0.9737], [0.08438277657432482, 0.9735], [0.08681158261597156, 0.9705], [0.07254934923574329, 0.9787], [0.08206492739962414, 0.9741], [0.07596589805902913, 0.9753], [0.080642052061297, 0.9764], [0.06888206332046538, 0.9779]]
# DR_modelB =[[0.03974304361072136, 0.9876], [0.03996831053330825, 0.9876], [0.03719625648912042, 0.9876], [0.04267445284897403, 0.9864], [0.0398399158939661, 0.9869], [0.04331016298498434, 0.9871], [0.04426171435196884, 0.9862], [0.04634853095332219, 0.9852], [0.040829259121021456, 0.9871], [0.03961709471888898, 0.9875]]
# DR_modelC =[[1.3572318864822388, 0.7461], [1.3761155730247498, 0.748], [1.3930660858154298, 0.7569], [1.393410537672043, 0.7507], [1.5211854327201844, 0.7323], [1.3560463187932967, 0.7508], [1.3079287543296814, 0.7495], [1.3462348965167998, 0.7565], [1.3506505819320678, 0.7513], [1.3838144593775272, 0.7545]]
# print(type(DR_modelA))








# LE_APath = []
# for i in range(10):
#   LE_APath.append(".\source_level\Label Error\mutated models\mutated_modelA\LeNet_label" + str(i+1) + ".hdf5")
# print(LE_APath)
#
# scores_A = []
# for each_path in LE_APath:
#     model = load_model(each_path)
#     score = model.evaluate(x_test, y_test, verbose=0)
#     scores_A.append(score)
# print(scores_A)
#
#
# LE_BPath = []
# for i in range(10):
#   LE_BPath.append(".\source_level\Label Error\mutated models\mutated_modelB\ModelB_label" + str(i+1) + ".hdf5")
# print(LE_BPath)
#
# scores_B = []
# for each_path in LE_BPath:
#     model = load_model(each_path)
#     score = model.evaluate(x_test, y_test, verbose=0)
#     scores_B.append(score)
# print(scores_B)
#
#
# LE_CPath = []
# for i in range(10):
#   LE_CPath.append(".\source_level\Label Error\mutated models\mutated_modelC\ModelC_label" + str(i+1) + ".hdf5")
# print(LE_CPath)
#
# scores_C = []
# for each_path in LE_CPath:
#     model = load_model(each_path)
#     score = model.evaluate(X_test, Y_test, verbose=0)
#     scores_C.append(score)
# print(scores_C)
# #[[0.22228772563934326, 0.97], [0.20888618611097337, 0.9737], [0.20760672714710235, 0.9712], [0.21376713494062424, 0.9693], [0.21770918315649032, 0.9713], [0.22432208193540573, 0.9708], [0.2235392827153206, 0.9725], [0.2135525136232376, 0.9756], [0.20568694179058075, 0.9717], [0.2113192256450653, 0.9711]]
#[[0.15547410079240798, 0.9836], [0.18722186522483825, 0.985], [0.17045741748809815, 0.983], [0.17046743659973146, 0.9864], [0.16408663023710252, 0.9864], [0.1694013414621353, 0.9864], [0.15579506888389588, 0.987], [0.15338357914686204, 0.985], [0.1522451644897461, 0.9869], [0.16200620820522307, 0.9852]]
#[[1.5364535426139831, 0.6742], [1.781445462989807, 0.6572], [1.7802992486953735, 0.6751], [1.617985811805725, 0.6767], [1.5293035984039307, 0.6742], [1.5808624273300171, 0.679], [1.6842700921058655, 0.6744], [1.7709633947372436, 0.6729], [1.5549330051422119, 0.6764], [1.5327579776763915, 0.682]]


LE_APath = []
for i in range(10):
  LE_APath.append(".\source_level\Data Missing\Mutated_model\modelA\DataMiss_ModelA" + str(i+1) + ".hdf5")
print(LE_APath)

scores_A = []
for each_path in LE_APath:
    model = load_model(each_path)
    score = model.evaluate(x_test, y_test, verbose=0)
    scores_A.append(score)
print(scores_A)


LE_BPath = []
for i in range(10):
  LE_BPath.append(".\source_level\Data Missing\Mutated_model\modelB\DataMiss_ModelB" + str(i+1) + ".hdf5")
print(LE_BPath)

scores_B = []
for each_path in LE_BPath:
    model = load_model(each_path)
    score = model.evaluate(x_test, y_test, verbose=0)
    scores_B.append(score)
print(scores_B)

LE_CPath = []
for i in range(10):
  LE_CPath.append(".\source_level\Data Missing\Mutated_model\modelC\DM_ModelC" + str(i+1) + ".hdf5")
print(LE_CPath)

scores_C = []
for each_path in LE_CPath:
    model = load_model(each_path)
    score = model.evaluate(X_test, Y_test, verbose=0)
    scores_C.append(score)
print(scores_C)

# [[0.08043353272611276, 0.9741], [0.09116187887862325, 0.9721], [0.08118307880889625, 0.975], [0.0802359853883274, 0.974], [0.08289926026463508, 0.9723], [0.09322201655767857, 0.9687], [0.08305447761723772, 0.9748], [0.08826395179592073, 0.9714], [0.09274025156795979, 0.9721], [0.07921511272536591, 0.9759]]
# [[0.04419708787742129, 0.9863], [0.043170666694029934, 0.9869], [0.04715632573570474, 0.9852], [0.04249719862082275, 0.9869], [0.04272867032898066, 0.9865], [0.05582114003588795, 0.9832], [0.04839648411634844, 0.985], [0.048748704815481325, 0.9855], [0.048619449243880807, 0.985], [0.04970013029706897, 0.983]]
# [[1.3632705677509307, 0.75], [1.3139490217685699, 0.7493], [1.4071769771575928, 0.7452], [1.3717220217704773, 0.7415], [1.5897764844894409, 0.725], [1.3209483644485474, 0.7384], [1.3843398851394653, 0.7272], [1.3927286358833313, 0.757], [1.302565113067627, 0.7495], [1.2641452778339386, 0.7511]]