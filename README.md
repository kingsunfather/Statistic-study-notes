# Statistic-study-notes
# 李航统计学习方法（第二版）的学习笔记，包括：  
## 1、每章重点数学公式的手动推导  
<br>均为手写然后扫描成图片，字迹不工整还望谅解，之后有时间会用Latex修正
-   [1.第一章数学公式推导](#1第一章数学公式推导)
    -   [1.1 极大似然估计推导](#11极大似然估计推导)
    -   [1.2 贝叶斯估计推导](#12贝叶斯估计推导)
    -   [1.3 利用Hoeffding推导泛化误差上界](#13利用Hoeffding推导泛化误差上界)
-   [2.第二章数学公式推导](#2第二章数学公式推导)
    -   [2.1 算法的收敛性证明Novikoff](#21算法的收敛性证明Novikoff)  
-   [3.第三章数学公式推导](#3第三章数学公式推导)    
    -   3.1 无数学推导，偏重算法实现-KNN  
-   [4.第四章数学公式推导](#4第四章数学公式推导)
    -   [4.1 用极大似然法估计朴素贝叶斯参数](#41用极大似然法估计朴素贝叶斯参数)
    -   [4.2 用贝叶斯估计法朴素贝叶斯参数](#42用贝叶斯估计法朴素贝叶斯参数)
    -   [4.3 证明后验概率最大化即期望风险最小化](#43证明后验概率最大化即期望风险最小化)
-   [5.第五章数学公式推导](#5第五章数学公式推导)
    -   5.1 无数学推导，偏重算法实现-决策树
-   [6.第六章数学公式推导](#6第六章数学公式推导)
    -   [6.1 最大熵模型的数学推导](#61最大熵模型的数学推导)
    -   [6.2 拉格朗日对偶性问题的数学推导](#62拉格朗日对偶性问题的数学推导)
    -   [6.3 改进的迭代尺度法数学推导](#63改进的迭代尺度法数学推导)
      <!-- /TOC -->  

## 2、每章算法的Python自实现    
[数据集为iris.csv（带Header)](https://github.com/kingsunfather/Statistic-study-notes/blob/master/codes/iris.csv)
### 第2章 感知机模型（使用Iris数据集）  
源代码[MyPerceptron.py](https://github.com/kingsunfather/Statistic-study-notes/blob/master/codes/MyPerceptron.py)
### 第3章 KNN模型（线性-使用Iris数据集 与 KD树-有点问题..修改后再上传）  
源代码[MyPerceptron.py](https://github.com/kingsunfather/Statistic-study-notes/blob/master/codes/MyKNN.py)
### 第4章 朴素贝叶斯模型（使用Iris数据集）
源代码[MyPerceptron.py](https://github.com/kingsunfather/Statistic-study-notes/blob/master/codes/MyNaiveBayes.py)
### 第5章 决策树模型（使用Iris数据集）
源代码[MyPerceptron.py](https://github.com/kingsunfather/Statistic-study-notes/blob/master/codes/MyDecisionTree.py)
### 第6章 逻辑斯提回归模型（使用Iris数据集，采用梯度下降方法）
源代码[MyPerceptron.py](https://github.com/kingsunfather/Statistic-study-notes/blob/master/codes/MyLogisticRegression.py)
### 第6章 最大熵模型(使用Iris数据集)
源代码[MyMaxEnt.py](https://github.com/kingsunfather/Statistic-study-notes/blob/master/codes/MyMaxEnt.py)

## 3、学习笔记汇总  
<br>学习笔记均为自己学习过程中记录在笔记本上然后拍照扫描成pdf
### [第1章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter1.pdf)
### [第2章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter2.pdf)
### [第3章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter3.pdf)  
### [第4章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter4.pdf)  
### [第5章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter5.pdf)  
### [第6章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter6.pdf)

## 4、每章节的课后习题实现   
<br>接下来每周都会定时更新课后习题的实现

## 1第一章数学公式推导

### 1.1极大似然估计推导  

![](/docImage/maximum_likelihood_estimation.jpg)   

### 1.2贝叶斯估计推导

![](/docImage/bayesian_estimation.jpg)  


### 1.3利用Hoeffding推导泛化误差上界  

![](/docImage/hoeffding1.jpg)  

![](/docImage/hoeffding2.jpg)  
 
## 2第二章数学公式推导  

### 2.1算法的收敛性证明Novikoff  

![](/docImage/Novikoff1.jpg)  
![](/docImage/Novikoff2.jpg)  
![](/docImage/Novikoff3.jpg)  

## 3第三章数学公式推导

## 4第四章数学公式推导

### 4.1用极大似然法估计朴素贝叶斯参数
![](/docImage/mle_naive_bayes.jpg)  

### 4.2用贝叶斯估计法朴素贝叶斯参数
![](/docImage/bayes_naive_bayes1.jpg)  
![](/docImage/bayes_naive_bayes2.jpg)  

### 4.3证明后验概率最大化即期望风险最小化
![](/docImage/poster_prob1.jpg)  
![](/docImage/poster_prob2.jpg)  

## 5第五章数学公式推导

## 6第六章数学公式推导

### 6.1最大熵模型的数学推导
![](/docImage/maximum_entropy1.jpg) 
![](/docImage/maximum_entropy2.jpg) 

### 6.2拉格朗日对偶性问题的数学推导
![](/docImage/lagrange_duality1.jpg) 
![](/docImage/lagrange_duality2.jpg) 
![](/docImage/lagrange_duality3.jpg) 

### 6.3改进的迭代尺度法数学推导
![](/docImage/iterative_method1.jpg) 
![](/docImage/iterative_method2.jpg) 
![](/docImage/iterative_method3.jpg) 











