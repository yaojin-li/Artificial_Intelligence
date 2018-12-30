"""
 !/usr/bin/env python3.6
 -*- coding: utf-8 -*-
 --------------------------------------
 @Description : 获得图片清晰度
 --------------------------------------
 @File        : getImgClarity.py
 @Time        : 2018/11/1 21:22
 @Software    : PyCharm
 --------------------------------------
 @Author      : lixj
 @Contact     : lixj_zj@163.com
 --------------------------------------
"""

import cv2

def getImageVar(imgPath):
    image = cv2.imread(imgPath)     #读取图片
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      #转换为灰度图
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()    #对图片用 3*3 的拉普拉斯算子做卷积，用以边缘检测
    return imageVar

if __name__ == '__main__':
    imgPath = "C:\\Users\\lenovo\\Desktop\\test.jpg"
    imageVar = getImageVar(imgPath)
    print(imageVar)