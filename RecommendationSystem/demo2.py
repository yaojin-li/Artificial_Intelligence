"""
 !/usr/bin/env python3.6
 -*- coding: utf-8 -*-
 --------------------------------
 Description :
 --------------------------------
 @Time    : 2018/10/31 10:29
 @File    : demo.py
 @Software: PyCharm
 --------------------------------
 @Author  : lixj
 @contact : lixj_zj@163.com
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

def init():
    user_file = "E:\\dataSet\\ml-100k\\u.user"
    data_file = "E:\\dataSet\\ml-100k\\u.data"
    item_file = "E:\\dataSet\\ml-100k\\u.item"

    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv(user_file, sep="|", names=u_cols, encoding='latin-1')
    print(users.__class__)

    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(data_file, sep='\t', names=r_cols,encoding='latin-1')

    i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown',
              'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western']
    items = pd.read_csv(item_file, sep='|', names=i_cols, encoding='latin-1')
    return users, ratings, items


def constructUserMovieMatrix(users, ratings):
    #构造用户电影矩阵
    num_users = users.user_id.unique().shape[0]   #用户数
    num_items = ratings.movie_id.unique().shape[0]   #电影总数
    data_matrix = np.zeros((num_users, num_items))
    for line in ratings.itertuples():
        data_matrix[line[1]-1, line[2]-1] = line[3]
    return data_matrix


def calculationSimilarity(data_matrix):
    # 转置计算商品之间相似度矩阵，不转置计算用户之间相似度矩阵
    user_similarity = cosine_similarity(data_matrix, dense_output=True)
    item_similarity = cosine_similarity(data_matrix.T, dense_output=True)
    return item_similarity


def rec_sys(items,ratings, item_similarity, keywords, k):
    movie_list = []
    movie_id = list(items[items['movie_title'].str.contains(keywords)].movie_id)[0]   # 获得电影的id
    movie_similarity = item_similarity[movie_id - 1] # 计算该电影的余弦相似度数组
    movie_similarity_index = np.argsort(-movie_similarity)[1:k + 1]     # 返回前k+1个最高相似度的索引位置

    for index in movie_similarity_index:
        rec_movie = list(items[items['movie_id'] == index + 1].movie_title) #电影名
        rec_movie.append(movie_similarity[index])    # 相似度
        rec_movie.append(ratings[ratings['movie_id'] == index+1].rating.mean()) # 平均评分
        rec_movie.append(len(ratings[ratings['movie_id'] == index+1]))    # 评分用户数
        movie_list.append(rec_movie)
    return movie_list


if __name__ == '__main__':
    beginTime = time.time()
    keywords = "mission"
    k = 5
    keywords = keywords.title()
    users, ratings, items = init()
    data_matrix = constructUserMovieMatrix(users, ratings)
    similarity = calculationSimilarity(data_matrix)
    movie_list = rec_sys(items, ratings, similarity, keywords, k)
    print(movie_list)
    print("耗时：", time.time()-beginTime)
