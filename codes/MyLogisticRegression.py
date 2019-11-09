#coding=utf-8
#Author:lrrlrr
#Email:kingsundad@gmail.com

import numpy as np
import time
import pandas as pd

#使用随机梯度下降
class LogisticRegression:
    def __init__(self,learn_rate=0.1,max_iter=10000,tol=1e-3):
        # 学习速率
        self.learn_rate=learn_rate
        # 迭代次数
        self.max_iter=max_iter
        # 迭代停止阈值
        self.tol=tol
        # 权重
        self.w=None

    def preprocessing(self,X):
        row=X.shape[0]
        #在末尾加上一列，数值为1
        y=np.ones(row).reshape(row, 1)
        X_prepro =np.hstack((X,y))
        return X_prepro

    def sigmod(self,x):
        return 1/(1+np.exp(-x))

    def train(self,X_train,y_train):
        X=self.preprocessing(X_train)
        y=y_train.T
        #初始化权重w
        self.w=np.array([[0]*X.shape[1]],dtype=np.float)
        i=0
        k=0
        for loop in range(self.max_iter):
            # 计算梯度
            z=np.dot(X[i],self.w.T)
            grad=X[i]*(y[i]-self.sigmod(z))
            # 利用梯度的绝对值作为迭代中止的条件
            if (np.abs(grad)<=self.tol).all():
                break
            else:
                # 更新权重w 梯度上升——求极大值
                self.w+=self.learn_rate*grad
                k+=1
                i=(i+1)%X.shape[0]
        print("迭代次数：{}次".format(k))
        print("最终梯度：{}".format(grad))
        print("最终权重：{}".format(self.w[0]))

    def predict(self,x):
        p=self.sigmod(np.dot(self.preprocessing(x),self.w.T))
        print("Y=1的概率被估计为：{:.2%}".format(p[0][0]))
        p[np.where(p>0.5)]=1
        p[np.where(p<0.5)]=0
        return p

    def cal_right_rate(self,X,y):
        y_c=self.predict(X)
        right_count = 0
        wrong_count = 0
        for i in range(len(y)):
            if y_c[i] == y[i]:
                right_count += 1
            else:
                wrong_count += 1
        return right_count / (right_count + wrong_count)
        error_rate=np.sum(np.abs(y_c-y.T))/y_c.shape[0]
        # return 1-error_rate

#根据文件路径读取Iris数据集数据
#return type: np.array
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
        if row["Species"] == "setosa" :
            y.append(1)
        else:
            y.append(0)
        X.append([float(row["Sepal.Length"]),float(row["Sepal.Width"]),float(row["Petal.Length"])])
    return np.array(X), np.array(y)

def main():
    star=time.time()
    # 训练数据集
    X, y = processData('iris.csv')
    X_train = X[0:149:30]
    y_train = y[0:149:30]

    #自己在数据集后面加上了干扰的实例
    X_test = X[0:151:1]
    y_test = y[0:151:1]

    # 构建实例，进行训练
    clf=LogisticRegression()
    clf.train(X_train,y_train)

    # 预测新数据
    y_predict=clf.predict(X_test)
    print("{}被分类为：{}".format(X_test[0],y_predict[0]))

    # 利用已有数据对训练模型进行评价
    correct_rate=clf.cal_right_rate(X_test,y_test)
    print("测试一共有{}组实例，正确率：{:.5%}".format(X_test.shape[0],correct_rate))
    end=time.time()
    print("用时：{:.5f}s".format(end-star))

if __name__=="__main__":
    main()