import keras
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn import metrics  # 模型评估
import random

def shuffle_set(data, label):
    
    train_row = list(range(len(label)))
    random.shuffle(train_row)
    Data = data[train_row]
    Label = label[train_row]
    return Data, Label

data = np.load('D:/RF_CNN/data/data_g5.npy')
label = np.load('D:/RF_CNN/data/label_g5.npy')
data = np.array(data)
label = np_utils.to_categorical(label, 2)
Data, Label = shuffle_set(data, label)

model = keras.models.load_model('D:/RF_CNN/compare/model2BN_g467_x.hdf5')# my model
loss, acc = model.evaluate(Data, Label, batch_size=512, verbose=2)
# loss1, loss2, loss3, acc1, acc2, acc3 = model.evaluate(Data, Label, batch_size=128, verbose=2)
# loss1, loss2,  acc1, acc2  = model.evaluate(Data, Label, batch_size=128, verbose=2)

F1 = []
Con_Matr = []

y_pred = model.predict(data)
y_test = np.argmax(label, axis=1)
y_pred = np.argmax(y_pred, axis=1)
f1 = metrics.f1_score(y_test, y_pred, average='macro')
F1.append(f1)
con_matr = confusion_matrix(y_test, y_pred)
Con_Matr.append(con_matr)
print(Con_Matr)
print(F1)