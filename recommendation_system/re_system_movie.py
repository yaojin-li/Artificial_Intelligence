"""
 !/usr/bin/env python3.6
 -*- coding: utf-8 -*-
 --------------------------------------
 @Description : 简单的电影推荐系统
 --------------------------------------
 @File        : re_system_movie.py
 @Time        : 2018/11/1 20:52
 @Software    : PyCharm
 --------------------------------------
 @Author      : lixj
 @Contact     : lixj_zj@163.com
 --------------------------------------
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time


def init():
    '''
    初始化，读入文件
    :return: 用户表，评分表，电影明细表
    '''
    user_file = "E:\\ZX_relatedResources\\recommendation_movie_ml-100k\\u.user"
    data_file = "E:\\ZX_relatedResources\\recommendation_movie_ml-100k\\u.data"
    item_file = "E:\\ZX_relatedResources\\recommendation_movie_ml-100k\\u.item"

    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv(user_file, sep="|", names=u_cols, encoding='latin-1')

    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(data_file, sep='\t', names=r_cols,encoding='latin-1')

    i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown',
              'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western']
    items = pd.read_csv(item_file, sep='|', names=i_cols, encoding='latin-1')
    return users, ratings, items


def constructUserMovieMatrix(users, ratings):
    '''
    构造用户-电影矩阵
    :param users: 用户表
    :param ratings: 打分表
    :return: 用户对电影评分的矩阵
    '''
    num_users = users.user_id.unique().shape[0]   #用户数
    num_items = ratings.movie_id.unique().shape[0]   #电影总数
    data_matrix = np.zeros((num_users, num_items))
    for line in ratings.itertuples():
        data_matrix[line[1]-1, line[2]-1] = line[3]
    return data_matrix


def calculationSimilarity(data_matrix):
    '''
    转置计算电影之间相似度矩阵，不转置计算用户之间相似度矩阵
    :param data_matrix: 评分矩阵
    :return: 电影之间的相似度矩阵
    '''
    user_similarity = cosine_similarity(data_matrix, dense_output=True)
    item_similarity = cosine_similarity(data_matrix.T, dense_output=True)
    return item_similarity


def rec_sys(items,ratings, item_similarity, keywords, k):
    '''
    推荐系统
    :param items: 电影明细表
    :param ratings: 评分表
    :param item_similarity: 电影相似度矩阵
    :param keywords: 输入的电影名称或关键字
    :param k: 推荐个数
    :return: 推荐电影结果列表
    '''
    movie_list = []     # 存储推荐电影结果列表
    movie_id = list(items[items['movie_title'].str.contains(keywords)].movie_id)[0]   # 获得电影的id
    movie_similarity = item_similarity[movie_id - 1]    # 计算该电影的余弦相似度数组
    movie_similarity_index = np.argsort(-movie_similarity)[1:k + 1]     # 返回前k+1个最高相似度的索引位置

    for index in movie_similarity_index:
        rec_movie = list(items[items['movie_id'] == index + 1].movie_title)     # 电影名
        rec_movie.append(movie_similarity[index])    # 相似度
        rec_movie.append(ratings[ratings['movie_id'] == index+1].rating.mean()) # 平均评分
        rec_movie.append(len(ratings[ratings['movie_id'] == index+1]))    # 评分用户数
        movie_list.append(rec_movie)
    return movie_list


if __name__ == '__main__':
    beginTime = time.time()
    keywords = "Assassins"
    k = 5
    keywords = keywords.title()
    users, ratings, items = init()
    data_matrix = constructUserMovieMatrix(users, ratings)
    similarity = calculationSimilarity(data_matrix)
    movie_list = rec_sys(items, ratings, similarity, keywords, k)
    print(movie_list)
    print("推荐耗时：", time.time()-beginTime)
