from keras.utils import np_utils
from sklearn.metrics import confusion_matrix  # 混淆矩阵
from sklearn.model_selection import train_test_split  # 划分数据集
import math
import numpy as np
import random

from sklearn import metrics  # 模型评估
from keras import Input
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Concatenate, Activation, BatchNormalization
from keras import backend as K
from keras.models import Model
from keras_flops import get_flops

for i in range(1, 6):

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
    data = np.load('D:\RF_CNN\data/data_g' + str(i) + 'g67.npy')
    label = np.load('D:\RF_CNN\data/label_g' + str(i) + 'g67.npy')
    label = np_utils.to_categorical(label, 2)
    Data, Label = shuffle_set(data, label)
    X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=0.000000000000001, random_state=32)

    def fire_Block(input, c1, c2, c3):

        conv1_0 = Conv1D(filters=c1, kernel_size=1, strides=1)(input)
        conv1_0 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1_0)
        conv1_0 = Activation('relu')(conv1_0)

        conv1_1 = Conv1D(filters=c2, kernel_size=1, strides=1)(conv1_0)
        conv1_1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1_1)
        conv1_1 = Activation('relu')(conv1_1)

        conv1_2 = Conv1D(filters=c3, kernel_size=3, strides=1, padding='same')(conv1_0)
        conv1_2 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1_2)
        conv1_2 = Activation('relu')(conv1_2)

        conv1_3 = Concatenate()([conv1_1, conv1_2])
        # conv1_3 = Activation('relu')(conv1_3)

        return conv1_3

    def models(input_shape):

        x = Conv1D(filters=96, kernel_size=7, strides=2)(input_shape)
        x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=3, strides=2)(x)

        x = fire_Block(x, c1=16, c2=64, c3=64)
        x = fire_Block(x, c1=16, c2=64, c3=64)
        x = fire_Block(x, c1=32, c2=128, c3=128)
        x = MaxPooling1D(pool_size=3, strides=2)(x)

        x = fire_Block(x, c1=32, c2=128, c3=128)
        x = fire_Block(x, c1=48, c2=192, c3=192)
        x = fire_Block(x, c1=48, c2=192, c3=192)
        x = fire_Block(x, c1=64, c2=256, c3=256)
        x = MaxPooling1D(pool_size=3, strides=2)(x)
        x = fire_Block(x, c1=64, c2=256, c3=256)

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

    filepath = "D:/RF_CNN/compare/squeezenet-BN_g" +str(i) + "67_x.hdf5"  # 保存模型的路径

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


