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
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Add, Multiply, \
    GlobalAveragePooling1D, Concatenate, AvgPool1D, BatchNormalization, ELU, Activation, DepthwiseConv1D,Conv2D, Lambda,\
    Reshape,GlobalMaxPooling1D,GRU
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
data = np.load('D:\RF_CNN/data/data_g5.npy')
label = np.load('D:\RF_CNN/data/label_g5.npy')
label = np_utils.to_categorical(label, 2)
Data, Label = shuffle_set(data, label)
X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=0.000000000000001, random_state=32)

D = 32
S = 16
C = D+S

# a = 0.7
# b = 0.2
# c = 0.1

r1 = 4
r2 = 2

def DC_Block1(input, k, c, padding='same'):

    conv1_1 = DepthwiseConv1D(kernel_size=k, strides=1, padding=padding)(input)
    conv1_1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1_1)
    conv1_1 = ELU()(conv1_1)

    # conv1_1 = Conv1D(filters=c, kernel_size=1, strides=1)(conv1_1)
    # conv1_1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1_1)
    # conv1_1 = ELU()(conv1_1)
    conv1_1 = MaxPooling1D(pool_size=2, strides=2)(conv1_1)

    return conv1_1

def Block1(input, k, c, padding='same'):

    conv1_1 = Conv1D(filters=c, kernel_size=k, strides=1, padding=padding)(input)
    conv1_1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1_1)
    conv1_1 = ELU()(conv1_1)
    conv1_1 = MaxPooling1D(pool_size=2, strides=2)(conv1_1)
    return conv1_1

def senet(inputs, c, r):

    x = GlobalAveragePooling1D()(inputs)
    x = Dense(int(x.shape[-1]) // r, activation='relu')(x)
    x = Dense(c, activation='sigmoid')(x)
    return x

def models(input_shape):

    x0 = Conv1D(filters=D, kernel_size=1, strides=1)(input_shape)
    x0 = BatchNormalization(momentum=0.99, epsilon=0.001)(x0)
    x0 = ELU()(x0)

    # y0 = Conv1D(filters=S, kernel_size=1, strides=1)(input_shape)
    # y0 = BatchNormalization(momentum=0.99, epsilon=0.001)(y0)
    # y0 = ELU()(y0)

    # 1
    x1 = DC_Block1(x0, k=3, c=D)
    y1 = Block1(input_shape, k=3, c=S)

    z1 = senet(x1, c=S, r=r1)
    z2 = senet(y1, c=D, r=r2)

    x1 = Multiply()([x1, z2])
    y1 = Multiply()([y1, z1])

    # 2
    x2 = DC_Block1(x1, k=3, c=D)
    y2 = Block1(y1, k=3, c=S)

    z1 = senet(x2, c=S, r=r1)
    z2 = senet(y2, c=D, r=r2)

    x2 = Multiply()([x2, z2])
    y2 = Multiply()([y2, z1])

    # 3
    x3 = DC_Block1(x2, k=3, c=D)
    y3 = Block1(y2, k=3, c=S)

    z1 = senet(x3, c=S, r=r1)
    z2 = senet(y3, c=D, r=r2)

    x3 = Multiply()([x3, z2])
    y3 = Multiply()([y3, z1])

    # 4
    x4 = DC_Block1(x3, k=3, c=D)
    y4 = Block1(y3, k=3, c=S)

    z1 = senet(x4, c=S, r=r1)
    z2 = senet(y4, c=D, r=r2)

    x4 = Multiply()([x4, z2])
    y4 = Multiply()([y4, z1])

    # 5
    x5 = DC_Block1(x4, k=3, c=D)
    y5 = Block1(y4, k=3, c=S)

    z1 = senet(x5, c=S, r=r1)
    z2 = senet(y5, c=D, r=r2)

    x5 = Multiply()([x5, z2])
    y5 = Multiply()([y5, z1])

    # 6
    x6 = DC_Block1(x5, k=3, c=D)
    y6 = Block1(y5, k=3, c=S)

    z1 = senet(x6, c=S, r=r1)
    z2 = senet(y6, c=D, r=r2)
    x6 = Multiply()([x6, z2])
    y6 = Multiply()([y6, z1])

    # 7
    x7 = DC_Block1(x6, k=3, c=D)
    y7 = Block1(y6, k=3, c=S)

    z1 = senet(x7, c=S, r=r1)
    z2 = senet(y7, c=D, r=r2)

    x7 = Multiply()([x7, z2])
    y7 = Multiply()([y7, z1])

    # 8
    x8 = DC_Block1(x7, k=3, c=D)
    y8 = Block1(y7, k=3, c=S)

    z1 = senet(x8, c=S, r=r1)
    z2 = senet(y8, c=D, r=r2)

    x8 = Multiply()([x8, z2])
    y8 = Multiply()([y8, z1])

    # 9
    x9 = DC_Block1(x8, k=3, c=D)
    y9 = Block1(y8, k=3, c=S)

    z1 = senet(x9, c=S, r=r1)
    z2 = senet(y9, c=D, r=r2)

    x9 = Multiply()([x9, z2])
    y9 = Multiply()([y9, z1])

    s1 = Concatenate()([x1, y1])
    s1 = GlobalAveragePooling1D()(s1)
    s1 = Dense(C, activation='sigmoid')(s1)

    s2 = Concatenate()([x2, y2])
    s2 = GlobalAveragePooling1D()(s2)
    s2 = Dense(C, activation='sigmoid')(s2)

    s3 = Concatenate()([x3, y3])
    s3 = GlobalAveragePooling1D()(s3)
    s3 = Dense(C, activation='sigmoid')(s3)

    c1 = Concatenate()([s1, s2, s3])
    c1 = Reshape((3, C, 1), input_shape=(None, 3 * C))(c1)
    c1 = Conv2D(filters=8, kernel_size=(3, 1), strides=1)(c1)
    c1 = BatchNormalization(momentum=0.99, epsilon=0.001)(c1)
    c1 = ELU()(c1)
    c1 = Flatten()(c1)

    s4 = Concatenate()([x4, y4])
    s4 = GlobalAveragePooling1D()(s4)
    s4 = Dense(C, activation='sigmoid')(s4)

    s5 = Concatenate()([x5, y5])
    s5 = GlobalAveragePooling1D()(s5)
    s5 = Dense(C, activation='sigmoid')(s5)

    s6 = Concatenate()([x6, y6])
    s6 = GlobalAveragePooling1D()(s6)
    s6 = Dense(C, activation='sigmoid')(s6)

    c2 = Concatenate()([s4, s5, s6])
    c2 = Reshape((3, C, 1), input_shape=(None, 3 * C))(c2)
    c2 = Conv2D(filters=8, kernel_size=(3, 1), strides=1)(c2)
    c2 = BatchNormalization(momentum=0.99, epsilon=0.001)(c2)
    c2 = ELU()(c2)
    c2 = Flatten()(c2)

    s7 = Concatenate()([x7, y7])
    s7 = GlobalAveragePooling1D()(s7)
    s7 = Dense(C, activation='sigmoid')(s7)

    s8 = Concatenate()([x8, y8])
    s8 = GlobalAveragePooling1D()(s8)
    s8 = Dense(C, activation='sigmoid')(s8)

    s9 = Concatenate()([x9, y9])
    s9 = GlobalAveragePooling1D()(s9)
    s9 = Dense(C, activation='sigmoid')(s9)

    c3 = Concatenate()([s7, s8, s9])
    c3 = Reshape((3, C, 1), input_shape=(None, 3 * C))(c3)
    c3 = Conv2D(filters=8, kernel_size=(3, 1), strides=1)(c3)
    c3 = BatchNormalization(momentum=0.99, epsilon=0.001)(c3)
    c3 = ELU()(c3)
    c3 = Flatten()(c3)

    # 重新用卷积选择，不用赋权重
    out = Concatenate()([c1, c2, c3])
    out = Reshape((3, 384, 1),  input_shape=(None, 1152))(out)
    out = Conv2D(filters=1, kernel_size=(3, 3), strides=1)(out)
    out = BatchNormalization(momentum=0.99, epsilon=0.001)(out)
    out = ELU()(out)
    out = Flatten()(out)

    out = Dense(2, activation='softmax')(out)
    out = Model(inputs=[input_shape], outputs=[out], name="RF_CNN")
    return out

inputs = Input(shape=(3000, 1))

model = models(inputs)
model.summary()
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 6:.05} M")

# plot_model(model, to_file='D:/RF_CNN/model.png')#打印模型图
# sgd = optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)

model.compile(loss='binary_crossentropy',
             optimizer='Adam', metrics='accuracy')

filepath = "D:/RF_CNN/model_g5.hdf5"  # 保存模型的路径

checkpoint = ModelCheckpoint(filepath=filepath, verbose=2,
                             monitor='val_accuracy', mode='max')
# , save_best_only='True'
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

