from keras.utils import np_utils
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn.model_selection import train_test_split  # 划分数据集
import math
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn import metrics  # 模型评估
from keras import Input
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv1D, MaxPooling1D, Add, \
    GlobalAveragePooling1D, Concatenate, BatchNormalization, ELU, Activation, DepthwiseConv1D

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
        K.set_value(model.optimizer.lr, lr * 10)
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

def Ghost_Block(input, c):

    conv1_0 = Conv1D(filters=c, kernel_size=1, strides=1, padding='same')(input)
    # conv1_0 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1_0)
    # conv1_0 = Activation('relu')(conv1_0)

    conv1_1 = DepthwiseConv1D(kernel_size=3, strides=1, padding='same')(conv1_0)
    # conv1_1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1_1)
    # conv1_1 = Activation('relu')(conv1_1)

    conv1_2 = Concatenate()([conv1_0, conv1_1])
    return conv1_2

def Ghost_1(input, c1, c2):

    x1 = Ghost_Block(input, c1)
    x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)
    x1 = Activation('relu')(x1)

    x2 = Ghost_Block(x1, c2)
    x2 = BatchNormalization(momentum=0.99, epsilon=0.001)(x2)
    y = Add()([input, x2])
    return y

def Ghost_2(input, c1, c2):

    x1 = Ghost_Block(input, c1)
    x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)
    x1 = Activation('relu')(x1)

    x1 = DepthwiseConv1D(kernel_size=3, strides=2, padding='same')(x1)
    x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)

    x2 = Ghost_Block(x1, c2)
    x2 = BatchNormalization(momentum=0.99, epsilon=0.001)(x2)

    x3 = Conv1D(filters=2*c2, kernel_size=1, strides=1)(input)
    x3 = DepthwiseConv1D(kernel_size=3, strides=2, padding='same')(x3)
    # x3 = BatchNormalization(momentum=0.99, epsilon=0.001)(x3)
    # x3 = Activation('relu')(x3)

    y = Add()([x2, x3])
    return y

def models(input_shape):

    x = Conv1D(filters=16, kernel_size=3, strides=2)(input_shape)
    x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    x = Activation('relu')(x)

    x = Ghost_1(x, 16, 8)
    x = Ghost_2(x, 48, 12)

    x = Ghost_1(x, 72, 12)
    x = Ghost_2(x, 72, 20)

    x = Ghost_1(x, 120, 20)
    x = Ghost_2(x, 240, 40)

    x = Ghost_1(x, 200, 40)
    x = Ghost_1(x, 184, 40)
    x = Ghost_1(x, 184, 40)
    x = Conv1D(filters=112, kernel_size=1, strides=1, padding='same')(x)
    x = Ghost_1(x, 480, 56)
    x = Ghost_1(x, 672, 56)
    x = Ghost_2(x, 672, 80)

    # x = Ghost_1(x, 960, 80)
    # x = Ghost_1(x, 960, 80)
    # x = Ghost_1(x, 960, 80)
    # x = Ghost_1(x, 960, 80)
    x = Conv1D(filters=960, kernel_size=1, strides=1, padding='same')(x)

    out = GlobalAveragePooling1D()(x)
    out = Dense(2, activation='softmax')(out)
    out = Model(inputs=[input_shape], outputs=[out], name="RF_CNN")
    return out


inputs = Input(shape=(3000, 1))

model = models(inputs)
model.summary()
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 6:.05} M")

model.compile(loss='binary_crossentropy',
              optimizer='Adam', metrics='accuracy')

filepath = "D:/RF_CNN/compare/ghostnet_g567_x.hdf5"  # 保存模型的路径

checkpoint = ModelCheckpoint(filepath=filepath, verbose=2,
                             monitor='val_accuracy', mode='max')

reduce_lr = LearningRateScheduler(scheduler)  # 学习率的改变
callback_lists = [checkpoint, reduce_lr]

train_history = model.fit(x=X_train,
                          y=y_train, validation_split=0.3, verbose=2,
                          class_weight=None, callbacks=callback_lists,
                          epochs=32, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)  # 修改损失函数

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

#
# def show_train_history(train_history, train, validation):
#     plt.plot(train_history.history[train])
#     plt.plot(train_history.history[validation])
#     plt.title('Train History')
#     plt.ylabel(train)
#     plt.xlabel('Epoch')
#     plt.legend(['train', 'validation'], loc='lower right')
#     plt.show()
#
#
# show_train_history(train_history, 'accuracy', 'val_accuracy')  # 绘制准确率执行曲线
# show_train_history(train_history, 'loss', 'val_loss')  # 绘制损失函数执行曲线
#
# from sklearn.metrics import roc_curve  # 画roc曲线
# from sklearn.metrics import auc  # auc值计算
#
# y_pred_keras = y_pred
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
# auc_keras = auc(fpr_keras, tpr_keras)
#
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'r-.')
# plt.plot(fpr_keras, tpr_keras, '--', label='newNet (area = {:.4f})'.format(auc_keras))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='lower right')
# plt.show()

