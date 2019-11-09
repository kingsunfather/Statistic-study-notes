#coding=utf-8
#Author:lrrlrr
#Email:kingsundad@gmail.com

import numpy as np
import pandas as pd

#根据文件路径读取Iris数据集数据
#return type: list
def processData(filePath):
    print('开始读取数据')
    # 存放数据集的list，X表示的是输入可能有很多维度，y表示输出的分类只有一个
    X = []
    y = []
    #默认读取csv的头部
    df = pd.read_csv(filePath)
    #利用数据集合的第一个维度特征分类
    #遍历pandas中df的每一行
    for index, row in df.iterrows():
        if(row["Sepal.Length"]>=5.5) :
            y.append(1)
        else:
            y.append(-1)
        X.append([float(row["Sepal.Width"]),float(row["Petal.Length"])])
    return X, y


#感知机类
class MyPerceptron:
    def __init__(self):
        # 参数w
        self.w = None
        # 偏置b
        self.b = 0
        # 表示学习速率
        self.l_rate = 0.0001
        #表示迭代次数
        self.iter = 100

    #训练
    def train(self, X_train, y_train):
        print('开始训练')
        # 将数据转换成矩阵形式
        # 转换后的数据中每一个样本的向量都是横向的
        X_trainMat = np.mat(X_train)
        y_trainMat = np.mat(y_train).T
        # 获取数据矩阵的大小，为m*n
        m, n = np.shape(X_trainMat)
        #np.shape(X_trainMat)[1]表示的维度=样本的长度
        self.w = np.zeros((1, np.shape(X_trainMat)[1]))

        # 进行iter次迭代计算
        for k in range(self.iter):
            ##利用随机梯度下降
            for i in range(m):
                # 获取当前样本的向量
                xi = X_trainMat[i]
                # 获取当前样本所对应的标签
                yi = y_trainMat[i]
                # 判断是否是误分类样本
                # 误分类样本特诊为： -yi(w*xi+b)>=0，详细可参考书中2.2.2小节
                # 在书的公式中写的是>0，实际上如果=0，说明改点在超平面上，也是不正确的
                if -1 * yi * (self.w * xi.T + self.b) >= 0:
                    # 对于误分类样本，进行梯度下降，更新w和b
                    self.w = self.w + self.l_rate * yi * xi
                    self.b = self.b + self.l_rate * yi

    #测试
    def predict(self,X_test, y_test):
        print('开始预测')
        X_testMat = np.mat(X_test)
        y_testMat = np.mat(y_test).T

        #获取测试数据集矩阵的大小
        m, n = np.shape(X_testMat)
        #错误样本数计数
        rightCount = 0

        for i in range(m):
            #获得单个样本向量
            xi = X_testMat[i]
            #获得该样本标记
            yi = y_testMat[i]
            #获得运算结果
            result =  yi * (self.w * xi.T + self.b)
            #如果-yi(w*xi+b)>=0，说明该样本被误分类，错误样本数加一
            if result >= 0: rightCount += 1
        #正确率 = 1 - （样本分类错误数 / 样本总数）
        rightRate = rightCount / m
        #返回正确率
        return rightRate


def main():
    X,y = processData('iris.csv')

    # 构建感知机对象，对数据集训练并且预测
    perceptron=MyPerceptron()
    perceptron.train(X[0:100],y[0:100])
    rightRate = perceptron.predict(X[101:140],y[101:140])
    print('对测试集的分类的正确率为：',rightRate)
    #有二维输入，所以应该有2个w
    print('模型的参数w为：',perceptron.w)
    print('模型的参数b为',perceptron.b)


if __name__ == '__main__':
    main()