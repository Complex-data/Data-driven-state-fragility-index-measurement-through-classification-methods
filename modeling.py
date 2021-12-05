# 机器学习与数据建模
# ...... #
from xgboost import XGBClassifier
from ngboost import NGBRegressor, NGBClassifier
import os
import pydotplus
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
# from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.contrib import rnn
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM, Dropout, Activation, TimeDistributed
from sklearn.preprocessing import OneHotEncoder
# from keras.layers import Embedding
# from keras.layers import Dense, Dropout, Activation
# from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import TensorBoard
# import numpy as np
# import pandas as pd
# import pydotplus
# from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
# # 分类报告
# from sklearn.metrics import classification_report
# from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPool2D, Flatten
# from keras.layers.convolutional import Conv1D, Conv2D
# # 优化器
# from keras.optimizers import SGD, Adam
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.utils import np_utils

from keras.datasets import mnist
# Sequential按顺序构成的模型
from keras.models import Sequential
# Dense全连接层
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPool2D, Flatten, LSTM, GRU, Embedding
# 优化器
from keras.optimizers import SGD, Adam
from model_svm_wrapper import ModelSVMWrapper

import warnings
warnings.filterwarnings("ignore")
os.environ['PATH'] += os.pathsep + 'D:/Graphviz2.38/bin/'


def modeling(features, label):
    # 数据集切分
    # 切分训练集、测试集、验证集
    from sklearn.model_selection import train_test_split
    f_v = features.values
    f_name = features.columns.values
    l_v = label.values.astype('int')
    # test_size 测试集所占比例
    # 训练集：测试集：验证集 = 8.1：1:0.9
    # validation---0.1
    # X_tt, X_validation, Y_tt, Y_validation = train_test_split(
    #     f_v, l_v, test_size=0.2)
    # # train---0.6,test---0.2
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X_tt, Y_tt, test_size=0.25)
    X_train = features.head(488)
    Y_train = label.head(488)
    X_test = features.tail(122)
    Y_test = label.tail(122)

    print(X_train)

    # 衡量指标
    # 引入衡量指标
    # 预测值和实际值比较
    # accuracy_score---准确率,召回率---recall_score,F值---f1_score
    from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
    # 分类报告
    from sklearn.metrics import classification_report

    # KNN
    # NearestNeighbors---获取一个点最近的几个点
    from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
    # 朴素贝叶斯
    # 高斯朴素贝叶斯或伯努利朴素贝叶斯
    # 条件：特征必须是离散的，二值离散用伯努利更好 连续值将二值化
    # 在离散值下表现更好
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    # 决策树
    from sklearn.tree import DecisionTreeClassifier, export_graphviz

    # SVM支持向量机
    from sklearn.svm import SVC
    # 随机森林
    from sklearn.ensemble import RandomForestClassifier
    # Adaboost
    from sklearn.ensemble import AdaBoostClassifier
    # Logistic Regreesion
    from sklearn.linear_model import LogisticRegression
    # GBDT提升树
    from sklearn.ensemble import GradientBoostingClassifier
    # Xgboost
    from xgboost import XGBClassifier

    # 模型
    models = []

    # 加入KNN模型
    # n_neighbors---相邻的点数
    models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
    # # 加入高斯朴素贝叶斯
    models.append(('GaussianNB', GaussianNB()))
    # # 加入伯努利朴素贝叶斯
    models.append(('BernoulliNB', BernoulliNB()))
    # # 加入决策树
    # # Gini
    models.append(('DecisionTree', DecisionTreeClassifier()))
    # # 加入SVM
    # C为错分点的惩罚度 控制分类准确性 C越大 运算时间越长
    models.append(('SVM Classifier', SVC(C=100000)))
    # # ID3
    models.append(
        ('DecisionTreeEntropy', DecisionTreeClassifier(criterion='entropy')))

    # 加入随机森林
    # n_estimators--决策树的个数(默认=10), criterion--决策树使用的方法(默认为gini),
    # max_features--决策树使用的特征个数(默认=sqrt)--int 个数,float 比例,auto、sqrt 均为sqrt函数,
    # log2 以2为底求对数,None 全特征
    # bootstrap 是否有放回抽样 True-放回
    # oob_score 是否用未被采用的样本评估准确性
    models.append(('RandomForest', RandomForestClassifier(
        n_estimators=50, max_features=None, bootstrap=True)))
    # n_estimators=50, max_features=None, bootstrap=True
    # # 加入Adaboost
    # # base_estimator--进行弱分类的基本分类器
    # # n_estimators--分类器的数量(默认50)
    # models.append(('Adaboost', AdaBoostClassifier(n_estimators=100)))
    # # 加入逻辑回归
    # # penalty--l1 or l2 使用l1或l2正则化
    # # tol--精度为多少停止计算
    # # C--值越小，正则化因子比例越大
    # # solver--使用方法:'newton-cg','lbfgs','liblinear','sag','saga'
    # # max_iter 迭代次数
    models.append(('LogisticRegreesion', LogisticRegression(C=100000, tol=1e-10)))
    # 加入GBDT提升树
    models.append(('GBDT', GradientBoostingClassifier(
        max_depth=5, max_features='sqrt', n_estimators=100)))
    # # 加入xgboost
    models.append(('XGboost', XGBClassifier(
        activation='tanh',
        alpha=0.001, batch_size=16, hidden_layer_sizes=(40, 20, 10), max_iter=500, solver='adam')))

    # print(np.array(Y_test)
    data_name = ['Train:', 'Test:', 'DecisionTree']
    # 遍历模型
    for clf_name, clf in models:
        score = []
        corrs = []

        # xy_lst = [(X_train, Y_train), (X_test, Y_test)]
        # print(clf_name)
        for i in range(1):
            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            acc = accuracy_score(Y_test, Y_pred)
            df = pd.DataFrame()

            df.insert(0, 'test', Y_test)
            df.insert(1, 'predict', Y_pred)
            corr = df.corr()['predict']['test']
            score.append(acc)
            corrs.append(corr)

            # print(data_name[i])
            # target_names = ['class 0', 'class 1', 'class 2']
            # print(clf_name, '--ACC:', accuracy_score(Y_part, Y_pred))
            # print(clf_name, '--RES:', precision_score(Y_part, Y_pred, average='micro'))
            # print(clf_name, '--RES:', recall_score(Y_part, Y_pred, average='micro'))
            # print(clf_name, '--F1:', f1_score(Y_part, Y_pred, average='micro'))
            # print(classification_report(
            #     Y_part, Y_pred, target_names=target_names))
            # dot_data = export_graphviz(clf, out_file=None, feature_names=f_name,
            #                          class_names=['Fragile', 'steady', 'stable'],
            #                          filled=True,
            #                          rounded=True,
            #                          special_characters=True)
            # graph = pydotplus.graph_from_dot_data(dot_data)
            # graph.write_pdf('dt_tree.pdf')
        # acc = accuracy_score(Y_part, Y_pred)  # 准确率
        # # print('acc:', acc)
        # df = pd.DataFrame()
        # df.insert(0, 'test', Y_test)
        # df.insert(1, 'predict', Y_pred)
        # test = np.array(Y_test)
        # pred = np.array(Y_pred)
        # print(np.array(Y_test))
        print(np.array(Y_test))
        print(np.mean(score), np.mean(corrs))
        # print(np.array(Y_test))Y_test
        print(np.array(Y_pred))
        pd.DataFrame(Y_pred).to_csv('RF_122_' + str(clf_name) + '.csv', header=None)
        # # print(clf.feature_importances_)

        # with open('result/SVM_raking.csv', 'w') as f:
        #     for i in range(len(test)):
        #         f.write(str(test[i]) + ',' + str(pred[i])+'\n')

        # file_model = open('result/model_122.txt', 'a')
        # file_model.write(str(clf_name)+'\t'+str(round(np.mean(score), 4)) +
        #                  '\t'+str(round(np.mean(corrs), 4))+'\n')

        # print(clf.feature_importances_)
        # # print(' ')


def model_NGBoost(features, label):
    from ngboost.distns import k_categorical

    clf_name = features.columns.values
    clf_len = len(clf_name)

    X_train = features.head(366)
    Y_train = label.head(366)
    X_test = features.tail(122)
    Y_test = label.tail(122)

    Y_train = [int(i-1) for i in np.array(Y_train)]
    Y_test = [int(i - 1) for i in np.array(Y_test)]

    # from keras.utils import np_utils
    # Y_train = np_utils.to_categorical(Y_train)
    # Y_test = np_utils.to_categorical(Y_test)
    from ngboost.scores import LogScore, CRPScore
    # print(X_train.shape, Y_train.shape)
    # xgb = XGBClassifier().fit(X_train, Y_train)
    score = []
    corr = []
    for i in range(1):
        print(clf_name, i)
        ngb = NGBClassifier(Dist=k_categorical(
            122), verbose=False)
        ngb.fit(X_train, Y_train)
        Y_pred = ngb.predict(X_test)
        # print(Y_pred)
        acc = accuracy_score(Y_test, Y_pred)
        df = pd.DataFrame()
        df.insert(0, 'test', Y_test)
        df.insert(1, 'predict', Y_pred)
        cor = df.corr()['predict']['test']
        score.append(acc)
        corr.append(cor)

    # pd.DataFrame(Y_pred).to_csv('RF_3.csv')
    print(clf_name, np.mean(score), np.mean(corr))
    print(Y_pred)

    # file_mlp = open('result/NGBoost_122.txt', 'a')
    # file_mlp.write(str(clf_name)+'\t'+str(round(np.mean(score), 4)) +
    #                '\t' + str(round(np.mean(corr), 4)) + '\n')


def model_CatBoost(features, label):

    clf_name = features.columns.values
    clf_len = len(clf_name)

    X_train = features.head(488)
    Y_train = label.head(488)
    X_test = features.tail(122)
    Y_test = label.tail(122)

    # Y_train = [int(i-1) for i in np.array(Y_train)]
    # Y_test = [int(i - 1) for i in np.array(Y_test)]
    Y_train = [int(i) for i in np.array(Y_train)]
    Y_test = [int(i) for i in np.array(Y_test)]

    from catboost import CatBoostClassifier
    score = []
    corr = []

    for i in range(1):
        print(clf_name, i)
        ngb = CatBoostClassifier()
        ngb.fit(X_train, Y_train)
        Y_pred = ngb.predict(X_test)
        # print(Y_pred)
        acc = accuracy_score(Y_test, Y_pred)
        df = pd.DataFrame()
        df.insert(0, 'test', Y_test)
        df.insert(1, 'predict', Y_pred)
        cor = df.corr()['predict']['test']
        score.append(acc)
        corr.append(cor)

    print(Y_test)
    print(np.array(Y_pred).reshape(-1, 122))
    pd.DataFrame(Y_pred).to_csv('CatBoost_122.csv', header=None, index=False)
    # print(ngb.feature_importances_)
    print(clf_name, np.mean(score), np.mean(corr))

    # file_mlp = open('result/CatBoost_122.txt', 'a')
    # file_mlp.write(str(clf_name)+'\t'+str(round(np.mean(score), 4)) +
    #                '\t'+str(round(np.mean(corr), 4))+'\n')


def model_CNN(features, label):
    # from keras import backend as K
    # K.set_image_dim_ordering('th')

    # from __future__ import print_function
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler

    clf_name = features.columns.values
    clf_len = len(clf_name)

    X_train_origin = features.head(488)
    Y_train_origin = label.head(488)
    X_test_origin = features.tail(122)
    Y_test_origin = label.tail(122)

    X_train = X_train_origin.values.reshape(-1, clf_len, 1, 1)
    X_test = X_test_origin.values.reshape(-1, clf_len, 1, 1)

    from keras.utils import np_utils

    # 序列长度 一共28行
    time_steps = clf_len
    # input_size = 1
    # 隐藏层 size 50
    cell_size = 100

    Y_train = np_utils.to_categorical(Y_train_origin)
    Y_test = np_utils.to_categorical(Y_test_origin)
    print(Y_test)
    print(Y_test.shape)

    model = Sequential()
    # 第一个卷积层
    model.add(
        Convolution2D(input_shape=(clf_len, 1, 1), filters=32,
                      kernel_size=5,
                      strides=1,
                      padding='same',
                      activation='relu'
                      )
    )
    model.add(
        MaxPool2D(
            pool_size=2, strides=2, padding='same'
        )
    )
    # 第二个卷积层
    model.add(
        # 64
        Convolution2D(
            16,
            5,
            strides=1,
            padding='same',
            activation='relu'
        )
    )
    model.add(
        MaxPool2D(
            2, 2, 'same'
        )
    )
    model.add(Flatten())
    # 1024
    model.add(
        Dense(1024, activation='relu')
    )
    model.add(
        Dropout(0.4)
    )
    model.add(
        Dense(123, activation='softmax')
    )
    print(model.summary())
    adam = Adam(lr=1e-3)

    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=32, epochs=10)
    loss, accuray = model.evaluate(X_test, Y_test)
    print('\test loss', loss)
    print('accuray', accuray)

    Y_pred = model.predict(X_test)
    y_pre = []
    for idx in Y_pred:
        line = np.argmax(idx)
        y_pre.append(int(line + 1))
    df = pd.DataFrame()
    df.insert(0, 'test', Y_test_origin)
    df.insert(1, 'predict', y_pre)
    corr = df.corr()['predict']['test']
    print(clf_len, accuray, corr)
    print(y_pre)
    print(Y_test_origin)

    # file_mlp = open('result/CNN_L122.txt', 'a')
    # file_mlp.write(str(clf_name)+'\t'+str(round(accuray, 4)) +
    #                '\t'+str(round(corr, 4))+'\n')


def model_GRU(features, label):
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    X_train = features.head(488)
    Y_train = label.head(488)
    X_test = features.tail(122)
    Y_test_orign = label.tail(122)
    clf_name = features.columns.values
    print(np.array(Y_test_orign))

    from sklearn import preprocessing

    num_units = 2
    nb_time_steps = 1
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    # print(Y_train)
    enc = OneHotEncoder()
    Y_train_enc = enc.fit_transform(Y_train.values.reshape(-1, 1)).toarray()
    Y_test_enc = enc.fit_transform(
        Y_test_orign.values.reshape(-1, 1)).toarray()

    print(Y_train_enc.shape)
    Y_train = Y_train_enc.reshape(
        -1, 1, Y_train_enc.shape[1])
    Y_test = Y_test_enc.reshape(-1, 1, Y_train_enc.shape[1])
    # X_train = np.expand_dims(X_train, -1)
    # X_test = np.expand_dims(X_test, -1)
    # print(X_train.shape)
    # print(Y_test.shape)

    # print(X_train)
    # print(Y_test)

    activation_function = 'softmax'
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    loss_function = 'categorical_crossentropy'
    # batch_size = 488
    num_epochs = 3000
    batch_size = 8
    CELL_SIZE = 10
    sgd = SGD(
        lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)

    regressor = Sequential()
    regressor.add(GRU(units=50, activation='relu', return_sequences=True,
                      input_shape=(X_train.shape[1], X_train.shape[2])))
    regressor.add(Dense(122))
    regressor.add(Activation('softmax'))
    print(regressor.summary())
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    regressor.compile(optimizer='adam', loss=loss_function,
                      metrics=['accuracy'])
    regressor.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs)

    # kfold = KFold(n_splits=10, shuffle=True, random_state=10)
    # results = cross_val_score(regressor, X_train, Y_train, cv=kfold)
    # print("Baseline: %.2f%% (%.2f%%)" %
    #       (results.mean()*100, results.std()*100))

    score = regressor.evaluate(X_test, Y_test, batch_size=batch_size)
    # print(score)
    Y_pred = regressor.predict(X_test)
    Y_rank = []
    for line in Y_pred:
        index = np.argmax(line) + 1
        Y_rank.append(index)
    print(Y_rank)

    # print(Y_test)
    # rank_pred = [sorted(Y_pred).index(values) for values in Y_pred]
    # rank_test = [sorted(np.array(Y_test)).index(values)
    #              for values in np.array(Y_test)]
    df = pd.DataFrame()
    df.insert(0, 'test', np.array(Y_test_orign))
    df.insert(1, 'predict', np.array(Y_rank))
    corr = df.corr()['predict']['test']
    print(score, corr)
    # test = np.array(Y_test_orign)
    # pred = np.array(Y_rank)
    # with open('result/RNN_raking.csv', 'w') as f:
    #     for i in range(len(test)):
    #         f.write(str(test[i]) + ',' + str(pred[i])+'\n')

    # file_mlp = open('result/GRU_122_3000.txt', 'a')
    # file_mlp.write(str(clf_name)+'\t'+str(round(score[1], 4)) +
    #                '\t'+str(round(corr, 4))+'\n')

    print(score, corr)


def model_RNN(features, label):
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    X_train = features.head(488)
    Y_train = label.head(488)
    X_test = features.tail(122)
    Y_test_orign = label.tail(122)
    clf_name = features.columns.values
    print(np.array(Y_test_orign))

    from sklearn import preprocessing
    # Y_train = (Y_train['FSI_rank2'] - Y_train['FSI_rank2'].min())/(Y_train['FSI_rank2'].max() - Y_train['FSI_rank2'].min())
    # Y_test = preprocessing.scale(Y_test_orign)
    # Y_train = preprocessing.scale(Y_train)
    # Y_test = pd.DataFrame(Y_test)
    # Y_train = pd.DataFrame(Y_train)

    num_units = 2
    nb_time_steps = 1
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    # print(Y_train)
    enc = OneHotEncoder()
    Y_train_enc = enc.fit_transform(Y_train.values.reshape(-1, 1)).toarray()
    Y_test_enc = enc.fit_transform(
        Y_test_orign.values.reshape(-1, 1)).toarray()

    print(Y_train_enc.shape)
    Y_train = Y_train_enc.reshape(
        -1, 1, Y_train_enc.shape[1])
    Y_test = Y_test_enc.reshape(-1, 1, Y_train_enc.shape[1])
    # X_train = np.expand_dims(X_train, -1)
    # X_test = np.expand_dims(X_test, -1)
    # print(X_train.shape)
    # print(Y_test.shape)

    # print(X_train)
    # print(Y_test)

    activation_function = 'softmax'
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    loss_function = 'categorical_crossentropy'
    # batch_size = 488
    num_epochs = 4000
    batch_size = 8
    CELL_SIZE = 10
    sgd = SGD(
        lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)

    regressor = Sequential()
    regressor.add(
        LSTM(units=10, activation=activation_function,
             input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))  # X_train.shape[2])
    # regressor.add(Dense(1))
    # regressor.add(Activation('sigmoid'))
    regressor.add(Dense(122))
    regressor.add(Activation('softmax'))
    print(regressor.summary())
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    regressor.compile(optimizer='adam', loss=loss_function,
                      metrics=['accuracy'])
    regressor.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs)

    # kfold = KFold(n_splits=10, shuffle=True, random_state=10)
    # results = cross_val_score(regressor, X_train, Y_train, cv=kfold)
    # print("Baseline: %.2f%% (%.2f%%)" %
    #       (results.mean()*100, results.std()*100))

    score = regressor.evaluate(X_test, Y_test, batch_size=batch_size)
    # print(score)
    Y_pred = regressor.predict(X_test)
    Y_rank = []
    for line in Y_pred:
        index = np.argmax(line) + 1
        Y_rank.append(index)
    print(Y_rank)

    # print(Y_test)
    # rank_pred = [sorted(Y_pred).index(values) for values in Y_pred]
    # rank_test = [sorted(np.array(Y_test)).index(values)
    #              for values in np.array(Y_test)]
    df = pd.DataFrame()
    df.insert(0, 'test', np.array(Y_test_orign))
    df.insert(1, 'predict', np.array(Y_rank))
    corr = df.corr()['predict']['test']

    # test = np.array(Y_test_orign)
    # pred = np.array(Y_rank)
    # with open('result/RNN_raking.csv', 'w') as f:
    #     for i in range(len(test)):
    #         f.write(str(test[i]) + ',' + str(pred[i])+'\n')

    file_mlp = open('result/RNN_3.txt', 'a')
    file_mlp.write(str(clf_name)+'\t'+str(round(score[1], 4)) +
                   '\t'+str(round(corr, 4))+'\n')

    print(score, corr)


def model_MLP(features, label):
    from sklearn.model_selection import train_test_split

    from sklearn.model_selection import StratifiedKFold  # 交叉验证

    from sklearn.model_selection import GridSearchCV
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
    X_train = features.head(488)
    Y_train = label.head(488)
    X_test = features.tail(122)
    Y_test = label.tail(122)

    # one-hot

    clf_name = features.columns.values

    # kflod = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)

    # parameters = {'hidden_layer_sizes': [(100, 80, 60, 40), (200, 150, 100, 50,), (40, 20, 10,), (10, 10, 10), (8, 8, 8,), (100,)],
    #               'activation': ('identity', 'logistic', 'tanh', 'relu'),
    #               'solver': ('lbfgs', 'sgd', 'adam'),
    #               'alpha': [0.1, 0.01, 0.001],
    #               'batch_size': [256, 128, 64, 32, 16, 8],
    #               #   'learning_rate' = ('constant', 'invscaling', 'adaptive'),
    #               'max_iter': [2000, 1000, 500, 200, 100]}
    scores = []
    corrs = []
    for i in range(5):
        clf = MLPClassifier(activation='tanh', alpha=0.001, batch_size=16,
                            hidden_layer_sizes=(40, 20, 10), max_iter=2000, solver='adam')
        result = clf.fit(X_train, Y_train)
        score = clf.score(X_test, Y_test)
        Y_pred = clf.predict(X_test)
        # print(Y_pred)
        df = pd.DataFrame()
        df.insert(0, 'test', Y_test)
        df.insert(1, 'predict', Y_pred)
        corr = df.corr()['predict']['test']
        scores.append(score)
        corrs.append(corr)
    print(np.mean(scores), np.mean(corrs))
    print(Y_pred)

    # file_mlp = open('result/MLP_122.txt', 'a')
    # file_mlp.write(str(clf_name)+'\t'+str(round(np.mean(scores), 4)) +
    #                '\t'+str(round(np.mean(corrs), 4))+'\n')


def model_RNN2(features, label):
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    X_train = features.head(488)
    Y_train = label.head(488)
    X_test = features.tail(122)
    Y_test_orign = label.tail(122)
    print(np.array(Y_test_orign))
    clf_name = features.columns.values

    from sklearn import preprocessing
    # Y_train = (Y_train['FSI_rank2'] - Y_train['FSI_rank2'].min())/(Y_train['FSI_rank2'].max() - Y_train['FSI_rank2'].min())
    # Y_test = preprocessing.scale(Y_test_orign)
    # Y_train = preprocessing.scale(Y_train)
    # Y_test = pd.DataFrame(Y_test)
    # Y_train = pd.DataFrame(Y_train)

    num_units = 2
    nb_time_steps = 2
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    # print(Y_train)
    enc = OneHotEncoder()
    Y_train_enc = enc.fit_transform(Y_train.values.reshape(-1, 1)).toarray()
    Y_test_enc = enc.fit_transform(
        Y_test_orign.values.reshape(-1, 1)).toarray()

    print(Y_train_enc.shape)
    Y_train = Y_train_enc.reshape(
        -1, 1, Y_train_enc.shape[1])
    Y_test = Y_test_enc.reshape(-1, 1, Y_train_enc.shape[1])
    # X_train = np.expand_dims(X_train, -1)
    # X_test = np.expand_dims(X_test, -1)
    # print(X_train.shape)
    # print(Y_test.shape)

    # print(X_train)
    # print(Y_test)

    activation_function = 'softmax'
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    loss_function = 'categorical_crossentropy'
    # batch_size = 488
    num_epochs = 4000
    batch_size = 8
    CELL_SIZE = 10
    sgd = SGD(
        lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)

    regressor = Sequential()
    regressor.add(
        LSTM(units=10, activation=activation_function,
             input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))  # X_train.shape[2])
    # regressor.add(Dense(1))
    # regressor.add(Activation('sigmoid'))
    regressor.add(Dense(122))
    regressor.add(Activation('softmax'))
    print(regressor.summary())
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    regressor.compile(optimizer='adam', loss=loss_function,
                      metrics=['accuracy'])
    regressor.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs)

    # kfold = KFold(n_splits=10, shuffle=True, random_state=10)
    # results = cross_val_score(regressor, X_train, Y_train, cv=kfold)
    # print("Baseline: %.2f%% (%.2f%%)" %
    #       (results.mean()*100, results.std()*100))

    score = regressor.evaluate(X_test, Y_test, batch_size=batch_size)
    # print(score)
    Y_pred = regressor.predict(X_test)
    Y_rank = []
    for line in Y_pred:
        index = np.argmax(line) + 1
        Y_rank.append(index)
    print(Y_rank)

    # print(Y_test)
    # rank_pred = [sorted(Y_pred).index(values) for values in Y_pred]
    # rank_test = [sorted(np.array(Y_test)).index(values)
    #              for values in np.array(Y_test)]
    df = pd.DataFrame()
    df.insert(0, 'test', np.array(Y_test_orign))
    df.insert(1, 'predict', np.array(Y_rank))
    corr = df.corr()['predict']['test']

    print(score, corr)

    # print(Y_test)
    # df1 = pd.DataFrame()
    # df1.insert(0, 'test', np.array(Y_test))
    # df1.insert(1, 'predict', np.array(Y_pred))
    # corr1 = df1.corr()['predict']['test']

    # print(score, corr)
    # file_rnn = open('result/RNN_122.txt', 'a')
    # file_rnn.write(str(clf_name)+'\t'+str(round(score[1], 4)) +
    #                '\t'+str(round(corr, 4))+'\n')
    test = np.array(Y_test_orign)
    pred = np.array(Y_rank)

    # print(test)
    print(pred)
    # with open('result/LSTM_raking.csv', 'w') as f:
    #     for i in range(len(test)):
    #         f.write(str(test[i]) + ',' + str(pred[i])+'\n')
    # # print(Y_pred)


def model_CNN_SVM(features, label):
    # from keras import backend as K
    # K.set_image_dim_ordering('th')

    # from __future__ import print_function
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler

    clf_name = features.columns.values
    clf_len = len(clf_name)
    X_train_origin = features.head(488)
    Y_train_origin = label.head(488)
    X_test_origin = features.tail(122)
    Y_test_origin = label.tail(122)

    X_train = X_train_origin.values.reshape(-1, clf_len, 1, 1)
    X_test = X_test_origin.values.reshape(-1, clf_len, 1, 1)

    from keras.utils import np_utils

    Y_train = np_utils.to_categorical(Y_train_origin)
    Y_test = np_utils.to_categorical(Y_test_origin)

    # print(Y_train.shape)
    # print(X_train.shape)

    model = Sequential()
    model.add(Convolution2D(input_shape=(clf_len, 1, 1), filters=32,
                            kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=1, padding='same'))
    # 第二个卷积层
    model.add(Convolution2D(16, 5, strides=1,
                            padding='same', activation='relu'))
    model.add(MaxPool2D(2, 2, 'same'))
    model.add(Flatten())
    # 1024
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5, name="split_layer"))
    # model.add(Flatten(name="intermediate_output"))
    model.add(Dense(123, activation='softmax'))
    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # import datetime
    tensorboard_callback = 0
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir="./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)

    print(model.summary())

    wrapper = ModelSVMWrapper(model)

    accuracy = {
        "with_svm": [],
        "without_svm": []
    }

    epochs = 1
    for i in range(epochs):
        print('Starting run: {}'.format(i))
        wrapper.fit(X_train, Y_train, X_train_origin,
                    Y_train_origin, clf_len, 16, 300, tensorboard_callback)
        accuracy["with_svm"].append(wrapper.evaluate(X_test, Y_test_origin)[0])
        # train-data
        wrapper.transform(X_train, X_test, Y_train_origin, Y_test_origin)
    print(accuracy["with_svm"])

    # print(accuracy["with_svm"])

# model.fit(X_train, Y_train, batch_size=32, epochs=2000)
# loss, accuray = model.evaluate(X_test, Y_test)
# print('\test loss', loss)
# print('accuray', accuray)

# Y_pred = model.predict(X_test)
# y_pre = []
# for idx in Y_pred:
#     line = np.argmax(idx)
#     y_pre.append(int(line + 1))
# df = pd.DataFrame()
# df.insert(0, 'test', Y_test_origin)
# df.insert(1, 'predict', y_pre)
# corr = df.corr()['predict']['test']
# print(clf_len, accuray, corr)

# file_mlp = open('result/CNN_L122.txt', 'a')
# file_mlp.write(str(clf_name)+'\t'+str(round(accuray, 4)) +
#                '\t'+str(round(corr, 4))+'\n')
