from keras.utils import np_utils
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn.model_selection import train_test_split  # 划分数据集
import math
import numpy as np
import matplotlib.pyplot as plt
import random
# from rf_block import Block_64, Block_128, Block_64_1D_3, Block_64_1D_5, Block_128_1D, Block_64_3, Block_64_5
from DCRF import RFx64, RFx128, RFx256, TREEx64,TREEx128,DRFx64,DRFx128,ResDC_Block1,Res_Block1,Res_Block2,ResDC_Block2
from sklearn import metrics  # 模型评估
from keras import Input
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Add,Multiply,\
     GlobalAveragePooling1D, Concatenate, AvgPool1D, BatchNormalization,ELU,Activation, UpSampling1D
from keras import backend as K
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras_flops import get_flops

def shuffle_set(data, label):
    train_row = list(range(len(label)))
    random.shuffle(train_row)
    Data = data[train_row]
    Label = label[train_row]
    return Data, Label

# 学习率更新以及调整
def scheduler(epoch):
    if epoch == 0:
        lr = K.get_value(model.optimizer.lr)  # keras默认0.001
        K.set_value(model.optimizer.lr, lr*10)
        print("lr changed to {}".format(lr))
    if epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * math.pow(0.95, epoch))
        print("lr changed to {}".format(lr))
    return K.get_value(model.optimizer.lr)

# 定义空矩阵
F1 = []
Con_Matr = []
# 数据导入
data = np.load('D:\RF_CNN\data/data_g5g67.npy')
label = np.load('D:\RF_CNN\data/label_g5g67.npy')
label = np_utils.to_categorical(label, 2)
Data, Label = shuffle_set(data, label)
X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=0.000000000000001, random_state=32)

def models(input_shape):

    conv1 = Conv1D(filters=8, kernel_size=7, strides=1, padding='same')(input_shape)
    conv1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1)
    conv1 = Activation('relu')(conv1)

    conv1 = Conv1D(filters=16, kernel_size=7, strides=1, padding='same')(conv1)
    conv1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling1D(pool_size=2, strides=2)(conv1)

    conv1 = Conv1D(filters=32, kernel_size=7, strides=1, padding='same')(conv1)
    conv1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling1D(pool_size=2, strides=2)(conv1)

    conv1 = Conv1D(filters=64, kernel_size=7, strides=1, padding='same')(conv1)
    conv1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling1D(pool_size=2, strides=2)(conv1)

    conv1 = Conv1D(filters=128, kernel_size=7, strides=1, padding='same')(conv1)
    conv1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling1D(pool_size=2, strides=2)(conv1)

    conv1 = Conv1D(filters=1, kernel_size=7, strides=1, padding='same')(conv1)
    conv1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling1D(pool_size=3, strides=3)(conv1)

    conv2 = UpSampling1D(3)(conv1)
    conv2 = Conv1D(filters=128, kernel_size=7, strides=1, padding='same')(conv2)
    conv2 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = UpSampling1D(2)(conv2)
    conv2 = Conv1D(filters=64, kernel_size=7, strides=1, padding='same')(conv2)
    conv2 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = UpSampling1D(2)(conv2)
    conv2 = Conv1D(filters=32, kernel_size=7, strides=1, padding='same')(conv2)
    conv2 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = UpSampling1D(2)(conv2)
    conv2 = Conv1D(filters=16, kernel_size=7, strides=1, padding='same')(conv2)
    conv2 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = UpSampling1D(2)(conv2)
    conv2 = Conv1D(filters=8, kernel_size=7, strides=1, padding='same')(conv2)
    conv2 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = Conv1D(filters=1, kernel_size=7, strides=1, padding='same')(conv2)
    conv2 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv2)
    conv2 = Activation('relu')(conv2)

    out1 = Flatten()(conv1)
    out1 = Dense(32, activation='relu')(out1)
    out1 = Dense(2, activation='softmax')(out1)

    out2 = Flatten()(conv2)
    out2 = Dense(3000)(out2)

    out = Model(inputs=[input_shape], outputs=[out1, out2], name="RF_CNN")
    return out

inputs = Input(shape=(3000, 1))

model = models(inputs)
model.summary()
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 6:.05} M")

plot_model(model, to_file='D:/RF_CNN/model.png')#打印模型图

model.compile(loss=['binary_crossentropy', 'mean_squared_error'], loss_weights=[1, 1], optimizer='Adam', metrics='accuracy')

filepath = "D:/RF_CNN/compare/model3_g567_x.hdf5" # 保存模型的路径

checkpoint = ModelCheckpoint(filepath=filepath, verbose=2,
                             monitor='val_accuracy', mode='max')

reduce_lr = LearningRateScheduler(scheduler)  # 学习率的改变
callback_lists = [checkpoint, reduce_lr]

train_history = model.fit(x=X_train,
                          y=[y_train, X_train], validation_split=0.3, verbose=2,
                          class_weight=None, callbacks=callback_lists,
                          epochs=32, batch_size=32)

loss, accuracy = model.evaluate(X_test, [y_test, X_test])  # 修改损失函数

Acc = []
Loss = []
Acc.append(accuracy)  # 用append进行叠加
Loss.append(loss)

y_pred = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)
f1 = metrics.f1_score(y_test, y_pred, average='macro')
F1.append(f1)  # 用append叠加
con_matr = confusion_matrix(y_test, y_pred)
Con_Matr.append(con_matr)
print(Con_Matr)
print(F1)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()

show_train_history(train_history, 'accuracy', 'val_accuracy')  # 绘制准确率执行曲线
show_train_history(train_history, 'loss', 'val_loss')  # 绘制损失函数执行曲线

from sklearn.metrics import roc_curve#画roc曲线
from sklearn.metrics import auc#auc值计算

y_pred_keras = y_pred
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'r-.')
plt.plot(fpr_keras, tpr_keras,'--', label='newNet (area = {:.4f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()

