#coding=utf-8
#Author:lrrlrr
#Email:kingsundad@gmail.com

import numpy as np
import pandas as pd
import math
import random

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
        if row["Species"] == "setosa" :
            y.append(1)
        else:
            y.append(-1)
        X.append([float(row["Sepal.Length"]),float(row["Sepal.Width"]),float(row["Petal.Length"])])
    return np.array(X), np.array(y)


# X_train:训练数据集
# y_train: 训练测试集
# sigma: 高斯核中分母的σ,在核函数中σ的值,高度依赖样本特征值范围，特征值范围较大时若不相应增大σ会导致所有计算得到的核函数均为0
# C:软间隔中的惩罚参数,调和间隔与误分类点的系数
# toler:松弛变量
class SVM:
    def __init__(self, X_train, y_train, sigma = 10, C = 200, toler = 0.001):

        self.train_XMat = np.mat(X_train)
        # 训练标签集，为了方便后续运算提前做了转置，变为列向量
        self.train_yMat = np.mat(y_train).T
        # m：训练集数量    n：样本特征数目
        self.m, self.n = np.shape(self.train_XMat)
        self.sigma = sigma
        self.C = C
        self.toler = toler

        # 核函数（初始化时提前计算）
        self.k = self.calculateKernel()
        # SVM中的偏置b
        self.b = 0
        # α 长度为训练集数目
        self.alpha = [0] * self.train_XMat.shape[0]
        # SMO运算过程中的Ei
        self.E = [0 * self.train_yMat[i, 0] for i in range(self.train_yMat.shape[0])]
        self.supportVecIndex = []


# 使用高斯核函数
    def calculateKernel(self):
        #初始化高斯核结果矩阵 大小 = 训练集长度m * 训练集长度m
        #k[i][j] = Xi * Xj
        k = [[0 for i in range(self.m)] for j in range(self.m)]
        for i in range(self.m):
            X = self.train_XMat[i, :]
            for j in range(i, self.m):
                Z = self.train_XMat[j, :]
                #先计算||X - Z||^2
                result = (X - Z) * (X - Z).T
                #分子除以分母后去指数，得到高斯核结果
                result = np.exp(-1 * result / (2 * self.sigma**2))
                #将Xi*Xj的结果存放入k[i][j]和k[j][i]中
                k[i][j] = result
                k[j][i] = result
        return k

# 查看第i个α是否满足KKT条件
    def isSatisfyKKT(self, i):
        gxi =self.calculate_gxi(i)
        yi = self.train_yMat[i]
        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
                and (math.fabs(yi * gxi - 1) < self.toler):
            return True

        return False

    def calculate_gxi(self, i):
        gxi = 0
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        # 遍历每一个非零α，i为非零α的下标
        for j in index:
            #计算g(xi)
            gxi += self.alpha[j] * self.train_yMat[j] * self.k[j][i]
        # 求和结束后再单独加上偏置b
        gxi += self.b

        #返回
        return gxi

    def calculateEi(self, i):
        # 计算g(xi)
        gxi = self.calculate_gxi(i)
        # Ei = g(xi) - yi,直接将结果作为Ei返回
        return gxi - self.train_yMat[i]


# E1: 第一个变量的E1
# i: 第一个变量α的下标
    def getAlphaJ(self, E1, i):
        E2 = 0
        maxE1_E2 = -1
        maxIndex = -1
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]

        for j in nozeroE:
            E2_tmp = self.calculateEi(j)
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                #更新
                maxE1_E2 = math.fabs(E1 - E2_tmp)
                E2 = E2_tmp
                maxIndex = j
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                maxIndex = int(random.uniform(0, self.m))
            E2 = self.calculateEi(maxIndex)
        return E2, maxIndex

    def train(self, count = 100):
        countCur = 0; parameterChanged = 1
        while (countCur < count) and (parameterChanged > 0):
            countCur += 1
            parameterChanged = 0

            for i in range(self.m):
                #是否满足KKT条件，如果不满足则作为SMO中第一个变量从而进行优化
                if self.isSatisfyKKT(i) == False:
                    #如果下标为i的α不满足KKT条件，则进行优化
                    E1 = self.calculateEi(i)
                    E2, j = self.getAlphaJ(E1, i)

                    y1 = self.train_yMat[i]
                    y2 = self.train_yMat[j]

                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]

                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)

                    if L == H:
                        continue

                    #计算α的新值
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]

                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)

                    if alphaNew_2 < L: alphaNew_2 = L
                    elif alphaNew_2 > H: alphaNew_2 = H
                    #更新α1
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)

                    #计算b1和b2
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b

                    #依据α1和α2的值范围确定新b
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2

                    #将更新后的各类值写入，进行更新
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew

                    self.E[i] = self.calculateEi(i)
                    self.E[j] = self.calculateEi(j)

                    #如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
                    #反之则自增1
                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1

        #全部计算结束后，重新遍历一遍α，查找里面的支持向量
        for i in range(self.m):
            #如果α>0，说明是支持向量
            if self.alpha[i] > 0:
                #将支持向量的索引保存起来
                self.supportVecIndex.append(i)

# 单独计算核函数
    def calculateSinglKernel(self, x1, x2):
        # 计算高斯核
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.sigma ** 2))
        return np.exp(result)

# 对样本的标签进行预测
    def predict(self, x):
        result = 0
        for i in self.supportVecIndex:
            # 遍历所有支持向量，计算求和式
            tmp = self.calculateSinglKernel(self.train_XMat[i, :], np.mat(x))
            result += self.alpha[i] * self.train_yMat[i] * tmp
        # 偏置b
        result += self.b

        return np.sign(result)



    def test(self, X_test, y_test):

        rightCount = 0

        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                rightCount += 1
        return rightCount / len(X_test)


if __name__ == '__main__':

    X, y = processData('iris.csv')

    X_train = X[0:149:50]
    y_train = y[0:149:50]

    # 自己在数据集后面加上了干扰的实例
    X_test = X[0:150:1]
    y_test = y[0:150:1]

    # 初始化SVM类
    svm = SVM(X_train, y_train, 10, 200, 0.001)

    # 开始训练
    svm.train()

    # 开始测试
    rightRate = svm.test(X_test, y_test)
    print('准确率为百分之 %d' % (rightRate * 100))