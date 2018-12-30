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

user_file = "E:\\dataSet\\ml-100k\\u.data"
movie_file = "E:\\dataSet\\ml-100k\\u.item.txt" # 文件另存为 utf-8 类型

user_df = pd.read_csv(user_file, sep="\t", names=['user_id', 'item_id', 'rating', 'timestamp'])
movie_df = pd.read_csv(movie_file,
                       encoding='utf-8',
                       sep='|',
                       usecols=['movie_id', 'movie_title', 'release date'],
                       names=['movie_id', 'movie_title', 'release date', 'video release date',
                              'IMDb URL', 'unknown','Action', 'Adventure', 'Animation',
                              'children', 'Comedy', 'Crime','Documentary', 'Draama', 'Pantasy',
                              'File-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                              'Thriller', 'War', 'Western']
                       )

#
num_users = user_df.user_id.unique().shape[0]   #用户数
num_items = user_df.item_id.unique().shape[0]   #电影总数
# print(num_users, num_items)
data_matrix = np.zeros((num_users, num_items))
for line in user_df.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]
# print(data_matrix)

#
# 转置计算商品之间相似度，不转置计算用户之间相似度
item_similarity = cosine_similarity(data_matrix.T, dense_output=True)
# print(item_similarity)


#
def movie_recsys(keywords, k):
    keywords = keywords.title()
    print(keywords)
    movie_list = []
    #获得电影的id
    movie_id = list(movie_df[movie_df['movie_title'].str.contains(keywords)].movie_id)[0]
    #计算该电影的余弦相似度数组
    movie_similarity = item_similarity[movie_id-1]
    print(movie_similarity)
    #返回前k+1一个最高相似度的索引位置
    movie_similarity_index = np.argsort(-movie_similarity)[:k+1]
    print(np.argsort(movie_similarity))
    print(movie_similarity_index)

    # try:
    #     movieid = list(movie_df[movie_df['movie_title'].str.contains(keywords)].movie_id)[0]
    #     movie_similarity = item_similarity[movieid-1]
    #     movie_similarity_index = np.argsort(-movie_similarity)[:k+1]
    #
    #     for i in movie_similarity_index:
    #         print(i)
    #         # rec_movies = list(item_df[item_df.movie_id==(i+1)].movie_title)
    #         # rec_movies.append(movie_similarity[i])
    #         # rec_movies.append(len(df[df.item_id==(i+1)]))
    # except:
    #     print("not found")
    return movie_list


if __name__ == '__main__':
    # name = input("输入电影名或关键词：")
    # num = input("数量：")
    name = "mission"
    num = 5
    result = movie_recsys(name, num)
    # print(result)


