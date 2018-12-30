"""
 !/usr/bin/env python3.6
 -*- coding: utf-8 -*-
 --------------------------------------
 @Description : 朴素贝叶斯分类
 --------------------------------------
 @File        : naiveBayes.py
 @Time        : 2018/8/25 22:10
 @Software    : PyCharm
 --------------------------------------
 @Author      : lixj
 @Contact     : lixj_zj@163.com
 --------------------------------------
"""

from numpy import *


def loadDataSet():
    """
    创建模拟数据集
    英文文章通过分割成单词筛选侮辱性词汇；中文文章通过分词、打标签进行筛选与分类
    : return 每篇文章的单词列表wordListOfInfo，每篇文章对应的类别classVecOfInfo
    """
    article = [
        'my dog has flea problems help please',
        'maybe not take him to dog park stupid',
        'my dalmation is so cute I love him',
        'stop posting stupid worthless garbage',
        'mr licks ate my steak how to stop him',
        'quit buying worthless dog food stupid'
    ]
    wordListOfInfo = [word.split(" ") for word in article]
    classVecOfInfo = [0, 1, 0, 1, 0, 1]   # 列表长度等于单词矩阵数，其中的1代表对应的文件是侮辱性文章，0代表不是侮辱性文章
    return wordListOfInfo, classVecOfInfo


def getAllWordList(dataSet):
    '''
    获取所有单词的集合
    :param dataSet: 待过滤的数据集
    :return: 无重复的单词列表
    '''
    allWordSet = set([])
    for data in dataSet:
        allWordSet = allWordSet | set(data) # 对于每个list转换为set，取交集
    return list(allWordSet)


def setOfWords2Vec(allWordList, inputDataSet):
    '''
    标记样本词汇表中出现的单词（置1）
    :param allWordList: 训练样本的词汇表
    :param inputDataSet: 输入数据集
    :return: 匹配列表
    '''
    vecList = [0] * len(allWordList)   # 创建一个和词汇表等长的0向量列表
    for word in inputDataSet:
        if word in allWordList:
            vecList[allWordList.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)  # ! 待改进点
    return vecList


def categoryTraining_old(statisticWordMat, classVecOfInfo):
    '''
    训练分类器-old
    :param statisticWordMat: 统计单词矩阵
    :param classVecOfInfo: 文章对应的分类的信息
    :return: 匹配列表
    '''
    numDoc = len(statisticWordMat) # 统计训练的文章个数
    numWordsPreDoc = len(statisticWordMat[0])   # 统计训练样本词汇表中的单词个数
    pOfAbusive = sum(classVecOfInfo) / numDoc    # 样本中侮辱性文章出现的概率（classVecOfInfo中1的个数除以总数）

    # 构造不同类别下的单词出现列表
    p0Num = zeros(numWordsPreDoc)
    p1Num = zeros(numWordsPreDoc)

    # 词汇表在不同类别下单词出现的总数
    p0SumOfWords = 0.0
    p1SumOfWords = 0.0

    for i in range(len(classVecOfInfo)):
        # 如果是侮辱性文章，对侮辱性文章的统计单词矩阵相加，并对所有侮辱性文章中出现的侮辱单词求和
        if classVecOfInfo[i] == 1:
            p0Num += statisticWordMat[i]
            p0SumOfWords += sum(statisticWordMat[i])
        else:
            p1Num += statisticWordMat[i]
            p1SumOfWords += sum(statisticWordMat[i])

    # 在侮辱性文章的前提下，每个单词出现的概率（[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]）
    p0Vect = p0Num / p0SumOfWords

    # 在正常文章的前提下，每个单词出现的概率（[P(F1|C2),P(F2|C2),P(F3|C2),P(F4|C2),P(F5|C2)....]）
    p1Vect = p1Num / p1SumOfWords

    return p0Vect, p1Vect, pOfAbusive



def categoryTraining(statisticWordMat, classVecOfInfo):
    '''
    训练分类器
    param: statisticWordMat 单词统计矩阵
    param: classVecOfInfo 文章对应的分类的信息
    return: 匹配列表
    '''
    numDoc = len(statisticWordMat) # 统计训练的文章个数
    numWordsPreDoc = len(statisticWordMat[0])   # 统计训练样本词汇表中的单词个数
    pOfAbusive = sum(classVecOfInfo) / float(numDoc)    # 样本中侮辱性文章出现的概率（classVecOfInfo中1的个数除以总数）

    # 构造不同类别下的单词出现列表（所有词的初始化次数为1）
    p0Num = ones(numWordsPreDoc)
    p1Num = ones(numWordsPreDoc)

    # 词汇表在不同类别下单词出现的总数
    p0SumOfWords = 2.0  # 正常的统计
    p1SumOfWords = 2.0  # 侮辱的统计

    for i in range(numDoc):
        # 如果是侮辱性文章，对侮辱性文章的统计单词矩阵相加，并对所有侮辱性文章中出现的侮辱单词求和
        if classVecOfInfo[i] == 1:
            p0Num += statisticWordMat[i]
            p0SumOfWords += sum(statisticWordMat[i])
        else:
            p1Num += statisticWordMat[i]
            p1SumOfWords += sum(statisticWordMat[i])

    # 在侮辱性文章的前提下，每个单词出现的概率（[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]）
    p0Vect = log(p0Num / p0SumOfWords)

    # 在正常文章的前提下，每个单词出现的概率（[log(P(F1|C2)),log(P(F2|C2)),log(P(F3|C2)),log(P(F4|C2)),log(P(F5|C2))....]）
    p1Vect = log(p1Num / p1SumOfWords)

    return p0Vect, p1Vect, pOfAbusive


def classifyBayes(statisticTestWordMat, p0Vect, p1Vect, pOfAbusive):
    """
    朴素贝叶斯分类函数
    param: statisticTestWordMat 测试集单词统计矩阵
    param: p0Vect 在侮辱性文章的前提下，每个单词出现的概率，矩阵
    param: p1Vect 在正常文章的前提下，每个单词出现的概率，矩阵
    param: pOfAbusive 侮辱性文章出现的概率
    return: 0 or 1   侮辱性文章 或 正常文章

    # 计算公式 log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 上面的计算公式，没有除以贝叶斯准则的公式的分母，也就是 P(w) （P(w) 指的是此文档在所有的文档中出现的概率）就进行概率大小的比较了，
    # 因为 P(w) 针对的是包含侮辱和非侮辱的全部文档，所以 P(w) 是相同的。
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
    """
    p0 = sum(statisticTestWordMat * p0Vect) + log(pOfAbusive)   # P(w|c1) * P(c1) ，即贝叶斯准则的分子
    p1 = sum(statisticTestWordMat * p1Vect) + log(1 - pOfAbusive)   # P(w|c0) * P(c0) ，即贝叶斯准则的分子·
    if p0 > p1:
        return 0
    else:
        return 1


if __name__ == '__main__':
    # 1.收集数据
    wordListOfInfo, classVecOfInfo = loadDataSet()
    # 2.建立单词集合
    allWordList = getAllWordList(wordListOfInfo)
    # 3.统计单词是否出现并创建统计单词矩阵（记录单词在训练样本词汇表中的位置（置1），没有则为0）
    statisticWordMat = []
    for word in wordListOfInfo:
        statisticWordMat.append(setOfWords2Vec(allWordList, word))
    # 4.训练数据
    p0Vect, p1Vect, pOfAbusive = categoryTraining(array(statisticWordMat), array(classVecOfInfo))
    # 5.测试数据
    testEntry = ['stupid', 'garbage']
    statisticTestWordMat = array(setOfWords2Vec(allWordList, testEntry))
    print(testEntry, "classified as: ", classifyBayes(statisticTestWordMat, p0Vect, p1Vect, pOfAbusive))
