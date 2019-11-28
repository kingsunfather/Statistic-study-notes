#coding=utf-8
#Author:lrrlrr
#Email:kingsundad@gmail.com


import numpy as np
import random
import math

# 通过服从高斯分布的随机函数来伪造数据集
# mean0: 高斯0的均值、
# sigma0: 高斯0的方差
# alpha0: 高斯0的系数

# mean1: 高斯1的均值
# sigma1: 高斯1的方差
# alpha1: 高斯1的系数
# 混合了两个高斯分布的数据

def processData(mean0, sigma0, mean1, sigma1, alpha0, alpha1):
    #定义数据集长度为1000
    length = 1000

    #初始化高斯分布，数据长度为length * alpha
    data0 = np.random.normal(mean0, sigma0, int(length * alpha0))
    data1 = np.random.normal(mean1, sigma1, int(length * alpha1))
    
    trainData = []
    trainData.extend(data0)
    trainData.extend(data1)
    
    #对总的数据集进行打乱
    random.shuffle(trainData)
    return trainData

# 根据高斯密度函数计算值
# 返回整个可观测数据集的高斯分布密度（向量形式）
def calculateGauss(trainDataArr, mean, sigmod):
    result = (1 / (math.sqrt(2 * math.pi) * sigmod**2)) * np.exp(-1 * (trainDataArr - mean) * (trainDataArr - mean) / (2 * sigmod**2))
    return result


def E(trainDataArr, alpha0, mean0, sigmod0, alpha1, mean1, sigmod1):
    gamma0 = alpha0 * calculateGauss(trainDataArr, mean0, sigmod0)
    gamma1 = alpha1 * calculateGauss(trainDataArr, mean1, sigmod1)

    sum = gamma0 + gamma1
    gamma0 = gamma0 / sum
    gamma1 = gamma1 / sum
    return gamma0, gamma1

def M(meano, mean1, gamma0, gamma1, trainDataArr):
    mean0_new = np.dot(gamma0, trainDataArr) / np.sum(gamma0)
    mean1_new = np.dot(gamma1, trainDataArr) / np.sum(gamma1)

    sigmod0_new = math.sqrt(np.dot(gamma0, (trainDataArr - meano)**2) / np.sum(gamma0))
    sigmod1_new = math.sqrt(np.dot(gamma1, (trainDataArr - mean1)**2) / np.sum(gamma1))

    alpha0_new = np.sum(gamma0) / len(gamma0)
    alpha1_new = np.sum(gamma1) / len(gamma1)

    return mean0_new, mean1_new, sigmod0_new, sigmod1_new, alpha0_new, alpha1_new


def EM(trainDataList, iter = 500):
    trainDataArr = np.array(trainDataList)

    alpha0 = 0.5
    mean0 = 0
    sigmod0 = 1
    alpha1 = 0.5
    mean1 = 1
    sigmod1 = 1

    count = 0
    while (count < iter):
        count = count+1
        # E步
        gamma0, gamma1 = E(trainDataArr, alpha0, mean0, sigmod0, alpha1, mean1, sigmod1)
        # M步
        mean0, mean1, sigmod0, sigmod1, alpha0, alpha1 = M(mean0, mean1, gamma0, gamma1, trainDataArr)
    return alpha0, mean0, sigmod0, alpha1, mean1, sigmod1

if __name__ == '__main__':
    alpha0 = 0.1
    mean0 = -4.0
    sigmod0 = 0.6

    alpha1 = 0.9
    mean1 = 2.2
    sigmod1 = 0.1

    #初始化数据集
    trainDataList = processData(mean0, sigmod0, mean1, sigmod1, alpha0, alpha1)

    #开始EM算法，进行参数估计
    alpha0, mean0, sigmod0, alpha1, mean1, sigmod1 = EM(trainDataList)

    print('用EM计算之后的数据为:')
    print('alpha0:%.1f, mean0:%.1f, sigmod0:%.1f, alpha1:%.1f, mean1:%.1f, sigmod1:%.1f' % (
        alpha0, mean0, sigmod0, alpha1, mean1, sigmod1
    ))


