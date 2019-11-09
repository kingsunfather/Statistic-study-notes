#coding=utf-8
#Author:lrrlrr
#Email:kingsundad@gmail.com

import numpy as np
import pandas as pd

class NaiveBayes():
    def __init__(self,lambda_):
        # 贝叶斯系数 取0时，即为极大似然估计
        self.lambda_=lambda_
        # y的（类型：数量）
        self.y_types_count=None
        # y的（类型：概率）
        self.y_types_proba=None
        # （xi 的编号,xi的取值，y的类型）：概率
        self.x_types_proba=dict()

    def fit(self,X_train,y_train):
        # y的所有取值类型
        self.y_types=np.unique(y_train)
        # 转化成pandas df 数据格式
        X=pd.DataFrame(X_train)
        y=pd.DataFrame(y_train)
        # y的（类型：数量）统计
        self.y_types_count=y[0].value_counts()
        # y的（类型：概率）计算
        self.y_types_proba=(self.y_types_count+self.lambda_)/(y.shape[0]+len(self.y_types)*self.lambda_)

        # （xi 的编号,xi的取值，y的类型）：概率的计算  - 遍历xi
        for idx in X.columns:
            # 选取每一个y的类型
            for j in self.y_types:
                # 选择所有y==j为真的数据点的第idx个特征的值，并对这些值进行（类型：数量）统计
                p_x_y=X[(y==j).values][idx].value_counts()
                # 计算（xi 的编号,xi的取值，y的类型）：概率
                for i in p_x_y.index:
                    self.x_types_proba[(idx,i,j)]=(p_x_y[i]+self.lambda_)/(self.y_types_count[j]+p_x_y.shape[0]*self.lambda_)

    def predict(self,X_new):
        res=[]
        # 遍历y的可能取值
        for y in self.y_types:
            # 计算y的先验概率P(Y=ck)
            p_y=self.y_types_proba[y]
            p_xy=1
            for idx,x in enumerate(X_new):
                # 计算P(X=(x1,x2...xd)/Y=ck)
                p_xy*=self.x_types_proba[(idx,x,y)]
            res.append(p_y*p_xy)
        for i in range(len(self.y_types)):
            print("[{}]对应概率：{:.2%}".format(self.y_types[i],res[i]))
        #返回最大后验概率对应的y值
        return self.y_types[np.argmax(res)]

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
        X.append([float(row["Sepal.Width"]),str(row["Species"])])
    return np.array(X), np.array(y)

def main():
    X, y = processData('iris.csv')
    X_train = X[0:149:4]
    y_train = y[0:149:4]


    clf=NaiveBayes(lambda_= 0.5)
    clf.fit(X_train,y_train)

    X_test=np.array([3.5,"setosa"])
    y_predict=clf.predict(X_test)
    print("{}被分类为:{}".format(X_test,y_predict))

if __name__=="__main__":
    main()