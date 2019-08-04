"""
 !/usr/bin/env python3.6
 -*- coding: utf-8 -*-
 --------------------------------
 Description :
 --------------------------------
 @Time    : 2019/8/4 14:00
 @File    : linear_regression.py
 @Software: PyCharm
 --------------------------------
 @Author  : lixj
 @contact : lixj_zj@163.com
"""

import numpy
import pandas as pd

df = pd.DataFrame(pd.read_excel('D:\\ZX\\temp\\train_data.xlsx', dtype=str))

df1=df.copy()
# 数据清洗
# 1. 填补 visitor_id 为空的缺省值，以特定值填充某一列的空值
df["visitor_id"] = df["visitor_id"].fillna(0)
# print(df["visitor_id"][:30])

#删除/选取某列含有特定数值的行
df1=df1[df1['page_title'].isin(['视频见证结果页'])]
print(df1)
