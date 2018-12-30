"""
 !/usr/bin/env python3.6
 -*- coding: utf-8 -*-
 --------------------------------------
 @Description : 
 --------------------------------------
 @File        : test_only.py
 @Time        : 2018/11/24 22:36
 @Software    : PyCharm
 --------------------------------------
 @Author      : lixj
 @Contact     : lixj_zj@163.com
 --------------------------------------
"""

from sklearn.datasets import load_boston

boston = load_boston()
print(boston.data[:10])


