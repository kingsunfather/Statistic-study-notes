#coding=utf-8
#Author:lrrlrr
#Email:kingsundad@gmail.com

import numpy as np

# 依据训练文本统计PI、A、B
def trainHMM(fileName):
    # B：词语的开头
    # M：一个词语的中间词
    # E：一个词语的结果
    # S：非词语，单个词
    statuDict = {'B':0, 'M':1, 'E':2, 'S':3}

    # 每个字只有四种状态，所以下方的各类初始化中大小的参数均为4
    PI = np.zeros(4)
    # 初始化状态转移矩阵A，涉及到四种状态各自到四种状态的转移，因为大小为4x4
    A = np.zeros((4, 4))
    # 初始化观测概率矩阵，分别为四种状态到每个字的发射概率
    B = np.zeros((4, 65536))
    fr = open(fileName, encoding='utf-8')

    for line in fr.readlines():
        curLine = line.strip().split()
        wordLabel = []
        #对每一个单词进行遍历
        for i in range(len(curLine)):
            #如果长度为1，则直接将该字标记为S，即单个词
            if len(curLine[i]) == 1:
                label = 'S'
            else:
                label = 'B' + 'M' * (len(curLine[i]) - 2) + 'E'
            #如果是单行开头第一个字，PI中对应位置加1,
            if i == 0: PI[statuDict[label[0]]] += 1
            for j in range(len(label)):
                B[statuDict[label[j]]][ord(curLine[i][j])] += 1
            wordLabel.extend(label)
        for i in range(1, len(wordLabel)):
            A[statuDict[wordLabel[i - 1]]][statuDict[wordLabel[i]]] += 1

    sum = np.sum(PI)

    for i in range(len(PI)):
        if PI[i] == 0:  PI[i] = -3.14e+100
        else: PI[i] = np.log(PI[i] / sum)

    for i in range(len(A)):
        sum = np.sum(A[i])
        for j in range(len(A[i])):
            if A[i][j] == 0: A[i][j] = -3.14e+100
            else: A[i][j] = np.log(A[i][j] / sum)

    for i in range(len(B)):
        sum = np.sum(len(B[i]))
        for j in range(len(B[i])):
            if B[i][j] == 0: B[i][j] = -3.14e+100
            else:B[i][j] = np.log(B[i][j] / sum)

    return PI, A, B

def processTrainData(fileName):
    textData = []
    fr = open(fileName, encoding='utf-8')
    for line in fr.readlines():
        #读到的每行最后都有一个\n，使用strip将最后的回车符去掉
        line = line.strip()
        textData.append(line)

    return textData

def participleTestData(textData, PI, A, B):
    retArtical = []
    for line in textData:
        delta = [[0 for i in range(4)] for i in range(len(line))]
        for i in range(4):
            delta[0][i] = PI[i] + B[i][ord(line[0])]
        psi = [[0 for i in range(4)] for i in range(len(line))]

        for t in range(1, len(line)):
            for i in range(4):
                tmpDelta = [0] * 4
                for j in range(4):
                    tmpDelta[j] = delta[t - 1][j] + A[j][i]
                maxDelta = max(tmpDelta)
                maxDeltaIndex = tmpDelta.index(maxDelta)
                delta[t][i] = maxDelta + B[i][ord(line[t])]
                psi[t][i] = maxDeltaIndex

        sequence = []
        i_opt = delta[len(line) - 1].index(max(delta[len(line) - 1]))
        sequence.append(i_opt)

        for t in range(len(line) - 1, 0, -1):
            i_opt = psi[t][i_opt]
            sequence.append(i_opt)

        sequence.reverse()
        curLine = ''
        for i in range(len(line)):
            curLine += line[i]
            if (sequence[i] == 3 or sequence[i] == 2) and i != (len(line) - 1):
                curLine += '|'
        retArtical.append(curLine)
    return retArtical

if __name__ == '__main__':

    # 依据人民日报数据集计算HMM参数：PI、A、B
    PI, A, B = trainHMM('MyHMMTrainData.txt')

    # 读取测试文章
    textData = processTrainData('MyHMMTestData.txt')

    # 打印原文
    for line in textData:
        print(line)

    # 分词
    partiArtical = participleTestData(textData, PI, A, B)

    # 打印结果
    print('分词结果：')
    for line in partiArtical:
        print(line)
