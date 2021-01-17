# Statistic-study-notes
# 李航统计学习方法（第二版）的学习笔记，包括：  
## 1、每章重点数学公式的手动推导  
<br>均为手写然后扫描成图片，字迹不工整还望谅解，之后有时间会用Latex修正  
点击数学公式没有出现图片的情况 需要搭梯子才可以在线预览到数学推导的图片...  

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
-   [7.第七章数学公式推导](#7第七章数学公式推导)
    -   [7.1 软间隔最大化对偶问题](#71软间隔最大化对偶问题)
    -   [7.2 证明最大间隔分离超平面存在唯一性](#72证明最大间隔分离超平面存在唯一性)
-   [8.第八章数学公式推导](#8第八章数学公式推导)
    -   [8.1 证明AdaBoost是前向分步加法算法的特例](#81证明AdaBoost是前向分步加法算法的特例)
    -   [8.2 证明AdaBoost的训练误差界](#82证明AdaBoost的训练误差界)
-   [9.第九章数学公式推导](#9第九章数学公式推导)  
    -   [9.1 EM算法的导出](#91EM算法的导出)  
    -   [9.2 用EM算法估计高斯模混合模型](#92用EM算法估计高斯模混合模型)  
-   [10.第十章数学公式推导](#10第十章数学公式推导)  
    -   [10.1 前向算法两个公式的证明](#101前向算法两个公式的证明)  
    -   [10.2 维特比算法推导](#102维特比算法推导)  
-   [11.第十一章数学公式推导](#11第十一章数学公式推导)  
    -   [11.1 条件随机场的矩阵形式推导](#111条件随机场的矩阵形式推导)  
    -   [11.2 牛顿法和拟牛顿法的推导](#112牛顿法和拟牛顿法的推导) 
    
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
### 第7章 SVM(使用Iris数据集)
源代码[MySVM.py](https://github.com/kingsunfather/Statistic-study-notes/blob/master/codes/MySVM.py)  
### 第8章 AdaBoost(使用Iris数据集)
源代码[MyAdaBoost.py](https://github.com/kingsunfather/Statistic-study-notes/blob/master/codes/MyAdaBoost.py)  
### 第9章 EM算法(使用自己随机生成的符合高斯分布的数据)
源代码[MyEM.py](https://github.com/kingsunfather/Statistic-study-notes/blob/master/codes/MyEM.py)  
### 第10章 HMM算法(使用人民日报语料库进行训练,对输入的文本进行分词，12.8前完成)
源代码[MyHMM.py](https://github.com/kingsunfather/Statistic-study-notes/blob/master/codes/MyHMM.py)  

## 3、学习笔记汇总  
<br>学习笔记均为自己学习过程中记录在笔记本上然后拍照扫描成pdf
### [第1章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter1.pdf)  
### [第2章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter2.pdf)  
### [第3章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter3.pdf)  
### [第4章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter4.pdf)  
### [第5章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter5.pdf)  
### [第6章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter6.pdf)  
### [第7章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter7.pdf)  
### [第8章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter8.pdf)  
### [第9章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter9.pdf)  
### [第10章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter10.pdf)  
### [第11章学习笔记](https://github.com/kingsunfather/Statistic-study-notes/blob/master/notes/chapter11.pdf)  


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

## 7第七章数学公式推导  

### 7.1软间隔最大化对偶问题
![](/docImage/Soft_interval_maximization_dual1.jpg) 
![](/docImage/Soft_interval_maximization_dual2.jpg) 
![](/docImage/Soft_interval_maximization_dual3.jpg) 

### 7.2证明最大间隔分离超平面存在唯一性
![](/docImage/Maximum_separation_hyperplane1.jpg) 
![](/docImage/Maximum_separation_hyperplane2.jpg) 
![](/docImage/Maximum_separation_hyperplane3.jpg) 
![](/docImage/Maximum_separation_hyperplane4.jpg) 

## 8第八章数学公式推导  

### 8.1证明AdaBoost是前向分步加法算法的特例  
![](/docImage/8_1_1.jpg) 
![](/docImage/8_1_2.jpg) 
![](/docImage/8_1_3.jpg) 

### 8.2 证明AdaBoost的训练误差界  
![](/docImage/8_2_1.jpg) 
![](/docImage/8_2_2.jpg) 
![](/docImage/8_2_3.jpg) 
![](/docImage/8_2_4.jpg) 

## 9第九章数学公式推导  
### 9.1 EM算法的导出  
![](/docImage/9_1_1.jpg) 
![](/docImage/9_1_2.jpg) 

### 9.2 用EM算法估计高斯模混合模型  
![](/docImage/9_2_1.jpg) 
![](/docImage/9_2_2.jpg) 
![](/docImage/9_2_3.jpg) 

## 10.第十章数学公式推导  

### 10.1 前向算法两个公式的证明  
![](/docImage/10_1_1.jpg) 

### 10.2 维特比算法推导  
![](/docImage/10_2_1.jpg) 

## 11.第十一章数学公式推导   
### 11.1 条件随机场的矩阵形式推导  
![](/docImage/11_1_1.jpg) 
![](/docImage/11_1_1.jpg) 
### 11.2 牛顿法和拟牛顿法的推导   
![](/docImage/11_2_1.jpg) 
![](/docImage/11_2_2.jpg) 
![](/docImage/11_2_3.jpg) 
    
    
    










