#coding=utf-8
#Author:lrrlrr
#Email:kingsundad@gmail.com

import numpy as np
import pandas as pd

#根据文件路径读取Iris数据集数据
# #return type: np.array
def processData(filePath):
    # 存放数据集的list，X表示的是输入可能有很多维度，y表示输出的分类只有一个
    X = []
    y = []
    #默认读取csv的头部
    df = pd.read_csv(filePath)
    #利用数据集合的第一个维度特征分类
    #遍历pandas中df的每一行
    for index, row in df.iterrows():
        if row["Species"] == "setosa":
            y.append(1)
        else:
            y.append(-1)
    #进行二值化处理，使得样本数据为0/1
        X.append([int(float(row["Sepal.Length"])>5.0),int(float(row["Sepal.Width"])>3.5),int(float(row["Petal.Length"])>1.4)])
    
    return np.array(X), np.array(y)

def createOneLayerBoostTree(X_train, y_train, D):
    #获得样本数目及特征数量
    m, n = np.shape(X_train)
    #该字典代表了一层提升树，用于存放当前层提升树的参数
    oneLowerNumayerBoostTree = {}
    #初始化分类误差率为1，即100%
    oneLowerNumayerBoostTree['errorRate'] = 1
    #对每一个特征进行遍历，寻找用于划分的最合适的特征
    for i in range(n):
        #因为特征已经经过二值化，只能为0和1，因此分切点为-0.5， 0.5， 1.5
        for division in [-0.5, 0.5, 1.5]:
            #规则为如下所示：
            #LowerNumowerNumSetOne：LowerNumow is one：小于某值得是1
            #UpperNumSetOne：UpperNumigh is one：大于某值得是1
            for rule in ['LowerNumSetOne', 'UpperNumSetOne']:
                #按照第i个特征，以值division进行切割，进行当前设置得到的预测和分类错误率
                Gx, e = calculate_e_Gx(X_train, y_train, i, division, rule, D)
                #如果分类错误率e小于当前最小的e，那么将它作为最小的分类错误率保存
                if e < oneLowerNumayerBoostTree['errorRate']:
                    # 分类错误率
                    oneLowerNumayerBoostTree['errorRate'] = e
                    # 最优划分点
                    oneLowerNumayerBoostTree['division'] = division
                    # 划分规则
                    oneLowerNumayerBoostTree['rule'] = rule
                    # 预测结果
                    oneLowerNumayerBoostTree['Gx'] = Gx
                    # 特征索引
                    oneLowerNumayerBoostTree['feature'] = i
    return oneLowerNumayerBoostTree

def createBoostTree(X_train, y_train, treeNum):
    #将数据和标签转化为数组形式
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    #获得训练集数量以及特征个数
    m, n = np.shape(X_train)
    #初始化D为1/N
    D = [1 / m] * m
    #初始化提升树列表，每个位置为一层
    tree = []
    #循环创建提升树
    for i in range(treeNum):
        #得到当前层的提升树
        currentTree = createOneLayerBoostTree(X_train, y_train, D)
        # 这边由于用的是Iris数据集，数据量过小，所以currentTree['errorRate']即误差分类率可能为0
        # 因此在最后加上了0.0001来避免除数为0的错误
        alpha = 1/2 * np.log((1 - currentTree['errorRate']) / (currentTree['errorRate']+0.0001))
        #获得当前层的预测结果，用于下一步更新D
        Gx = currentTree['Gx']
        D = np.multiply(D, np.exp(-1 * alpha * np.multiply(y_train, Gx))) / sum(D)
        currentTree['alpha'] = alpha
        tree.append(currentTree)
    return tree

# 前提：数据进行二值处理
def predict(x, division, rule, feature):
    if rule == 'LowerNumSetOne':
        LowerNum = 1
        UpperNum = -1
    else:
        LowerNum = -1
        UpperNum = 1

    if x[feature] < division:
        return LowerNum
    else:
        return UpperNum

def test(X_test, y_test, tree):
    rightCount = 0
    for i in range(len(X_test)):
        result = 0
        for currentTree in tree:
            division = currentTree['division']
            rule = currentTree['rule']
            feature = currentTree['feature']
            alpha = currentTree['alpha']
            result += alpha * predict(X_test[i], division, rule, feature)
        #预测结果取sign值，如果大于0 sign为1，反之为0
        if np.sign(result) == y_test[i]: 
            rightCount += 1
    #返回准确率
    return rightCount / len(X_test)

#计算分类错误率
def calculate_e_Gx(X_train, y_train, n, division, rule, D):
    #初始化分类误差率为0
    e = 0
    x = X_train[:, n]
    y = y_train
    train = []
    if rule == 'LowerNumSetOne':
        LowerNum = 1
        UpperNum = -1
    else:
        LowerNum = -1
        UpperNum = 1

    #遍历样本的特征
    for i in range(X_train.shape[0]):
        if x[i] < division:
            #如果小于划分点，则预测为LowerNum
            #如果设置小于division为1，那么LowerNum就是1，
            #如果设置小于division为-1，LowerNum就是-1
            train.append(LowerNum)
            #如果预测错误，分类错误率要加上该分错的样本的权值
            if y[i] != LowerNum:
                e += D[i]
        elif x[i] >= division:
            train.append(UpperNum)
            if y[i] != UpperNum:
                e += D[i]
    return np.array(train), e

if __name__ == '__main__':
    X, y = processData('iris.csv')

    X_train = X[0:149:50]
    y_train = y[0:149:50]

    # 自己在数据集后面加上了干扰的实例
    X_test = X[0:150:1]
    y_test = y[0:150:1]

    #创建提升树,最后一个参数代表的是公式的m，即多少个模型
    tree = createBoostTree(X_train, y_train, 5)

    #准确率测试
    rightRate = test(X_test, y_test, tree)
    print('分类正确率为:',rightRate * 100, '%')