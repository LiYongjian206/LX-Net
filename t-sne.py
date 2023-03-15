import keras
import numpy as np
import keras.backend as k
from keras.layers import Concatenate
import matplotlib.pyplot as plt
from keras.layers import Concatenate


data = np.load('D:/RF_CNN/data/data_g5.npy')#our data
data = np.array(data)

model = keras.models.load_model('D:/RF_CNN/model_g367.hdf5')

layer_name = 'concatenate'
layer_output = model.get_layer(layer_name).output
layer_input = model.input
output_func = k.function([layer_input], [layer_output]) # construct function
x_preproc = data
print(x_preproc.shape)

for i in range(0, 5000):

    if i == 0:
        outputN = output_func([x_preproc[i][None, ...]])[0]  # 获取某一层的输出
        outputN = np.array(outputN)
        x = outputN.reshape(1, 48*1500)#重塑
        N = x
    if i != 0:
        outputN = output_func([x_preproc[i][None, ...]])[0]#获取某一层的输出
        outputN = np.array(outputN)
        y = outputN.reshape(1, 48*1500)  # 重塑
        # print(y.shape)
        N = np.vstack((N, y))
        # print(N.shape)



for i in range(25000,30000):
    if i == 25000:
        outputAF = output_func([x_preproc[i][None, ...]])[0]  # 获取某一层的输出
        outputAF = np.array(outputAF)
        x = outputAF.reshape(1, 48*1500)  # 重塑
        AF = x
    if i != 25000:
        outputAF = output_func([x_preproc[i][None, ...]])[0]  # 获取某一层的输出
        outputAF = np.array(outputAF)
        y = outputAF.reshape(1, 48*1500)  # 重塑
        # print(outputAF.shape)
        AF = np.vstack((AF, y))
        # print(AF.shape)

N = np.array(N)
AF = np.array(AF)
NAF = Concatenate(axis=0)([N, AF])

print(N.shape)
print(AF.shape)
print(NAF.shape)
# x = np.save('D:/RF_CNN/' + 'NAF3', NAF)
x = np.save('D:/RF_CNN/' + 'n', N)
y = np.save('D:/RF_CNN/' + 'af', AF)
