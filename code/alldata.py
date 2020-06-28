"""
此文件用来调取数据集，数据集一共三份：小中大

author： kai
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class datas(object):
    """
    data_type:默认为1
    1. 导入小数据
    2. 导入中数据
    3. 导入大数据

    data_standard:默认为1
    1、最大最小方法
    2、标准化方法

    mini_data:
    读取小数据，为波士顿房价数据集
    """
    def __init__(self):
        # self.data_type
        self.data_standard = 1
        #self.choose_data()
        pass


    # def choose_data(self):
    #     if (self.data_type == 1):
    #         self.mini_data()
    #     elif (self.data_type == 2):
    #         self.medium_data()
    #     elif (self.data_type == 3):
    #         self.tremendous_data()



    def mini_data(self):
        # 读取数据文件
        data = pd.read_excel('minidata.xlsx', encoding='utf-8')
        # 获取特征和标签值，总共有13组标签和1组特征
        train = data.iloc[:, 0:2]  #
        target = data.iloc[:, 2]  # 获取标签

        scaler = StandardScaler()
        scaler.fit(train)
        train = scaler.transform(train)
        target = np.array(target)  # 将y_data转换成数组
        x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2)
        return x_train, x_test, y_train, y_test


    def medium_data(self):
        # 读取数据文件
        data = pd.read_excel('data.xlsx', encoding='utf-8')
        # 获取特征和标签值，总共有13组标签和1组特征
        train = data.iloc[:, :-1]  # 获取前13组特征
        target = data.iloc[:, -1]  # 获取标签

        if (self.data_standard == 1):# 数据归一化（最大最小方法）
            scaler = MinMaxScaler()
            scaler.fit(train)
            train = scaler.transform(train)  # 此时输出的x_data就是数组了
        elif (self.data_standard == 1):
            #数据归一化（标准化方法）
            scaler = StandardScaler()
            scaler.fit(train)
            train = scaler.transform(train)
        target = np.array(target)  # 将y_data转换成数组
        x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2)
        return x_train, x_test, y_train, y_test


    def tremendous_data(self): #相对大的数据
        # 读取数据文件
        data = pd.read_csv('kc_house_data.csv', encoding='utf-8')
        # 获取特征和标签值，总共有13组标签和1组特征
        train = data.iloc[:, 3:]  # 获取前13组特征
        target = data.iloc[:, 2]  # 获取标签

        scaler = StandardScaler()
        scaler.fit(train)
        train = scaler.transform(train)
        target = np.array(target)  # 将y_data转换成数组
        x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2)
        return x_train, x_test, y_train, y_test
