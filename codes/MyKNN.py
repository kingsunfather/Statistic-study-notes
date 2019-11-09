#coding=utf-8
#Author:lrrlrr
#Email:kingsundad@gmail.com

import numpy as np
import pandas as pd
from collections import Counter
from concurrent import futures
import heapq

class KNN:
    def __init__(self,X_train,y_train,k=3):
        # 所需参数初始化
        self.k=k
        self.X_train=X_train
        self.y_train=y_train

    def predict_single(self,X_test):
        # 计算与前k个样本点欧氏距离，距离取负值是把原问题转化为取前k个最大的距离
        dist_list=[(-np.linalg.norm(X_test-self.X_train[i],ord=2),self.y_train[i],i)
                   for i in range(self.k)]

        # 利用前k个距离构建堆
        heapq.heapify(dist_list)

        # 遍历计算与剩下样本点的欧式距离
        for i in range(self.k,self.X_train.shape[0]):
            dist_i=(-np.linalg.norm(X_test-self.X_train[i],ord=2),self.y_train[i],i)
           #进行下堆操作
            if dist_i[0]>dist_list[0][0]:
                heapq.heappushpop(dist_list,dist_i)
            # 若dist_i 比 dis_list的最小值小，堆保持不变，继续遍历
            else:
                continue
        y_list=[dist_list[i][1] for i in range(self.k)]
        #[-1,1,1,-1...]
        # 对上述k个点的分类进行统计
        y_count=Counter(y_list).most_common()
        #{1:n,-1:m}
        return y_count[0][0]

    # 用多线程提高效率
    def predict_many(self,X_test):
        # 导入多线程
        with futures.ProcessPoolExecutor(max_workers=10) as executor:
            # 建立多线程任务
            tasks=[executor.submit(self.predict_single,X_test[i]) for i in range(X_test.shape[0])]
            # 驱动多线程运行
            done_iter=futures.as_completed(tasks)
            # 提取结果
            res=[future.result() for future in done_iter]
        return res

    def cal_right_rate(self,res,y_test):
        right_count = 0
        wrong_count = 0
        for i in range(len(res)):
            if res[i] == y_test[i]:
                right_count += 1
            else:
                wrong_count += 1
        return right_count / (right_count+wrong_count)

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
    return np.array(X), np.array(y)


def main():
    #获取数据
    X, y = processData('iris.csv')
    X_train = X[0:149:4]
    y_train = y[0:149:4]

    X_test = X[0:149:10]
    y_test = y[0:149:10]

    # 不同的k对分类结果的影响
    for k in range(1,6,2):
        #构建KNN实例
        clf=KNN(X_train,y_train,k=k)
        #对测试数据进行分类预测
        y_predict=clf.predict_many(X_test)
        print("k={},被分类为：{}".format(k,y_predict))
        print("正确率为: ", clf.cal_right_rate(y_predict,y_test))

if __name__=="__main__":
    main()