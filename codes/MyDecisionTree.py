#coding=utf-8
#Author:lrrlrr
#Email:kingsundad@gmail.com

import numpy as np
import pandas as pd
import math
from collections import namedtuple

# 定义节点
# 孩子节点、分类特征的取值、节点内容、节点分类特征、标签
class Node(namedtuple("Node","children type content feature label")):
    def __repr__(self):
        return str(tuple(self))

#决策树
class DecisionTree():
    def __init__(self,method="info_gain_ratio"):
        self.tree=None
        self.method=method

    #计算经验熵
    def _experienc_entropy(self,X):

        # 统计每个取值的出现频率
        x_types_prob=X.iloc[:,0].value_counts()/X.shape[0]
        # 计算经验熵
        x_experienc_entropy=sum((-p*math.log(p,2) for p in x_types_prob))
        return x_experienc_entropy

    #计算条件熵
    def _conditinal_entropy(self,X_train,y_train,feature):
        # feature特征下每个特征取值数量统计
        x_types_count= X_train[feature].value_counts()
        # 每个特征取值频率计算
        x_types_prob = x_types_count / X_train.shape[0]
        # 每个特征取值下类别y的经验熵
        x_experienc_entropy=[self._experienc_entropy(y_train[(X_train[feature]==i).values]) for i in x_types_count.index]
        # 特征feature对数据集的经验条件熵
        x_conditinal_entropy=(x_types_prob.mul(x_experienc_entropy)).sum()
        return x_conditinal_entropy

    #计算信息增益
    def _information_gain(self,X_train,y_train,feature):
        return self._experienc_entropy(y_train)-self._conditinal_entropy(X_train,y_train,feature)

    #计算信息增益比
    def _information_gain_ratio(self,X_train,y_train,features,feature):
        index=features.index(feature)
        return self._information_gain(X_train,y_train,feature)/self._experienc_entropy(X_train.iloc[:,index:index+1])

    #选择分类特征
    def _choose_feature(self,X_train,y_train,features):
        if self.method=="info_gain_ratio":
            info=[self._information_gain_ratio(X_train,y_train,features,feature) for feature in features]
        elif self.method=="info_gain":
            info=[self._information_gain(X_train,y_train,feature) for feature in features]
        else:
            raise TypeError
        optimal_feature=features[np.argmax(info)]
        return optimal_feature

    #递归构造决策树
    def _built_tree(self,X_train,y_train,features,type=None):
        # 只有一个节点或已经完全分类，则决策树停止继续分叉
        if len(features)==1 or len(np.unique(y_train))==1:
            label=list(y_train[0].value_counts().index)[0]
            return Node(children=None,type=type,content=(X_train,y_train),feature=None,label=label)
        else:
            # 选择分类特征值
            feature=self._choose_feature(X_train,y_train,features)
            features.remove(feature)
            # 构建节点，同时递归创建孩子节点
            features_iter=np.unique(X_train[feature])
            children=[]
            for item in features_iter:
                X_item=X_train[(X_train[feature]==item).values]
                y_item=y_train[(X_train[feature]==item).values]
                children.append(self._built_tree(X_item,y_item,features,type=item))
            return Node(children=children,type=type,content=None,feature=feature,label=None)

    #进行剪枝
    def _prune(self):
        pass

    def fit(self,X_train,y_train,features):
        self.tree=self._built_tree(X_train,y_train,features)


    def _search(self,X_new):
        tree=self.tree
        # 若还有孩子节点，则继续向下搜索，否则搜索停止，在当前节点获取标签
        while tree.children:
            for child in tree.children:
                if X_new[tree.feature].loc[0]==child.type:
                    tree=child
                    break
        return tree.label

    def predict(self,X_new):
       return self._search(X_new)

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
            y.append("是setosa花")
        else:
            y.append("不是setosa花")
        X.append([float(row["Sepal.Length"]),float(row["Sepal.Width"]),float(row["Petal.Length"]),float(row["Petal.Width"])])
    return np.array(X), np.array(y)

def main():
    # 训练数据集
    features = ["萼片长", "萼片宽", "花瓣长", "花瓣宽"]

    X , y = processData('iris.csv')

    X_train = X[0:149:4]
    y_train = y[0:149:4]

    X_test = X[0:149:10]
    y_test = y[0:149:10]


    X_train = pd.DataFrame(X_train, columns=features)
    y_train = pd.DataFrame(y_train)
    # 训练,使用信息增益
    clf=DecisionTree(method="info_gain")
    clf.fit(X_train,y_train,features.copy())
    print('训练结束')

    X_new= pd.DataFrame(X_test, columns=features)
    y_predict=clf.predict(X_new)
    print(y_predict)

if __name__=="__main__":
    main()