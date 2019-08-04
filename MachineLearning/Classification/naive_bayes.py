"""
 !/usr/bin/env python3.6
 -*- coding: utf-8 -*-
 --------------------------------------
 @Description : 朴素贝叶斯分类
 --------------------------------------
 @File        : naive_bayes.py
 @Time        : 2018/8/25 22:10
 @Software    : PyCharm
 --------------------------------------
 @Author      : lixj
 @Contact     : lixj_zj@163.com
 --------------------------------------
"""

from numpy import *


def load_data_set():
    """
    创建模拟数据集
    英文文章通过分割成单词筛选侮辱性词汇；中文文章通过分词、打标签进行筛选与分类
    : return 每篇文章的单词列表word_list_of_info，每篇文章对应的类别class_vec_of_info
    """
    article = [
        'my dog has flea problems help please',
        'maybe not take him to dog park stupid',
        'my dalmation is so cute I love him',
        'stop posting stupid worthless garbage',
        'mr licks ate my steak how to stop him',
        'quit buying worthless dog food stupid'
    ]
    word_list_of_info = [word.split(" ") for word in article]
    class_vec_of_info = [0, 1, 0, 1, 0, 1]   # 列表长度等于单词矩阵数，其中的1代表对应的文件是侮辱性文章，0代表不是侮辱性文章
    return word_list_of_info, class_vec_of_info


def get_all_word_list(data_set):
    '''
    获取所有单词的集合
    :param data_set: 待过滤的数据集
    :return: 无重复的单词列表
    '''
    all_word_set = set([])
    for data in data_set:
        all_word_set = all_word_set | set(data) # 对于每个list转换为set，取交集
    return list(all_word_set)


def set_of_words_to_vec(all_word_list, inputdata_set):
    '''
    标记样本词汇表中出现的单词（置1）
    :param all_word_list: 训练样本的词汇表
    :param inputdata_set: 输入数据集
    :return: 匹配列表
    '''
    vec_list = [0] * len(all_word_list)   # 创建一个和词汇表等长的0向量列表
    for word in inputdata_set:
        if word in all_word_list:
            vec_list[all_word_list.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)  # ! 待改进点
    return vec_list


def category_training_old(statistic_word_mat, class_vec_of_info):
    '''
    训练分类器-old
    :param statistic_word_mat: 统计单词矩阵
    :param class_vec_of_info: 文章对应的分类的信息
    :return: 匹配列表
    '''
    num_doc = len(statistic_word_mat) # 统计训练的文章个数
    num_words_pre_doc = len(statistic_word_mat[0])   # 统计训练样本词汇表中的单词个数
    p_of_abusive = sum(class_vec_of_info) / num_doc    # 样本中侮辱性文章出现的概率（class_vec_of_info中1的个数除以总数）

    # 构造不同类别下的单词出现列表
    p0_num = zeros(num_words_pre_doc)
    p1_num = zeros(num_words_pre_doc)

    # 词汇表在不同类别下单词出现的总数
    p0_sum_of_words = 0.0
    p1_sum_of_words = 0.0

    for i in range(len(class_vec_of_info)):
        # 如果是侮辱性文章，对侮辱性文章的统计单词矩阵相加，并对所有侮辱性文章中出现的侮辱单词求和
        if class_vec_of_info[i] == 1:
            p0_num += statistic_word_mat[i]
            p0_sum_of_words += sum(statistic_word_mat[i])
        else:
            p1_num += statistic_word_mat[i]
            p1_sum_of_words += sum(statistic_word_mat[i])

    # 在侮辱性文章的前提下，每个单词出现的概率（[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]）
    p0_vect = p0_num / p0_sum_of_words

    # 在正常文章的前提下，每个单词出现的概率（[P(F1|C2),P(F2|C2),P(F3|C2),P(F4|C2),P(F5|C2)....]）
    p1_vect = p1_num / p1_sum_of_words

    return p0_vect, p1_vect, p_of_abusive



def category_training(statistic_word_mat, class_vec_of_info):
    '''
    训练分类器
    param: statistic_word_mat 单词统计矩阵
    param: class_vec_of_info 文章对应的分类的信息
    return: 匹配列表
    '''
    num_doc = len(statistic_word_mat) # 统计训练的文章个数
    num_words_pre_doc = len(statistic_word_mat[0])   # 统计训练样本词汇表中的单词个数
    p_of_abusive = sum(class_vec_of_info) / float(num_doc)    # 样本中侮辱性文章出现的概率（class_vec_of_info中1的个数除以总数）

    # 构造不同类别下的单词出现列表（所有词的初始化次数为1）
    p0_num = ones(num_words_pre_doc)
    p1_num = ones(num_words_pre_doc)

    # 词汇表在不同类别下单词出现的总数
    p0_sum_of_words = 2.0  # 正常的统计
    p1_sum_of_words = 2.0  # 侮辱的统计

    for i in range(num_doc):
        # 如果是侮辱性文章，对侮辱性文章的统计单词矩阵相加，并对所有侮辱性文章中出现的侮辱单词求和
        if class_vec_of_info[i] == 1:
            p0_num += statistic_word_mat[i]
            p0_sum_of_words += sum(statistic_word_mat[i])
        else:
            p1_num += statistic_word_mat[i]
            p1_sum_of_words += sum(statistic_word_mat[i])

    # 在侮辱性文章的前提下，每个单词出现的概率（[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]）
    p0_vect = log(p0_num / p0_sum_of_words)

    # 在正常文章的前提下，每个单词出现的概率（[log(P(F1|C2)),log(P(F2|C2)),log(P(F3|C2)),log(P(F4|C2)),log(P(F5|C2))....]）
    p1_vect = log(p1_num / p1_sum_of_words)

    return p0_vect, p1_vect, p_of_abusive


def classify_bayes(statistic_test_word_mat, p0_vect, p1_vect, p_of_abusive):
    """
    朴素贝叶斯分类函数
    param: statistic_test_word_mat 测试集单词统计矩阵
    param: p0_vect 在侮辱性文章的前提下，每个单词出现的概率，矩阵
    param: p1_vect 在正常文章的前提下，每个单词出现的概率，矩阵
    param: p_of_abusive 侮辱性文章出现的概率
    return: 0 or 1   侮辱性文章 或 正常文章

    # 计算公式 log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 上面的计算公式，没有除以贝叶斯准则的公式的分母，也就是 P(w) （P(w) 指的是此文档在所有的文档中出现的概率）就进行概率大小的比较了，
    # 因为 P(w) 针对的是包含侮辱和非侮辱的全部文档，所以 P(w) 是相同的。
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
    """
    p0 = sum(statistic_test_word_mat * p0_vect) + log(p_of_abusive)   # P(w|c1) * P(c1) ，即贝叶斯准则的分子
    p1 = sum(statistic_test_word_mat * p1_vect) + log(1 - p_of_abusive)   # P(w|c0) * P(c0) ，即贝叶斯准则的分子·
    if p0 > p1:
        return 0
    else:
        return 1


if __name__ == '__main__':
    # 1.收集数据
    word_list_of_info, class_vec_of_info = load_data_set()
    # 2.建立单词集合
    all_word_list = get_all_word_list(word_list_of_info)
    # 3.统计单词是否出现并创建统计单词矩阵（记录单词在训练样本词汇表中的位置（置1），没有则为0）
    statistic_word_mat = []
    for word in word_list_of_info:
        statistic_word_mat.append(set_of_words_to_vec(all_word_list, word))
    # 4.训练数据
    p0_vect, p1_vect, p_of_abusive = category_training(array(statistic_word_mat), array(class_vec_of_info))
    # 5.测试数据
    testEntry = ['stupid', 'garbage']
    statistic_test_word_mat = array(set_of_words_to_vec(all_word_list, testEntry))
    print(testEntry, "classified as: ", classify_bayes(statistic_test_word_mat, p0_vect, p1_vect, p_of_abusive))
