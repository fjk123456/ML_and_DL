# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:31:17 2019

@author: PC
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

#pandas加载数据
data=pd.read_excel(r"F:\rock_image\lbp_wenli.xlsx")
#数据的归一化
scaler=MinMaxScaler(feature_range=(0,1))
re_num=scaler.fit_transform(data)

#K的个数
kmeans=KMeans(n_clusters=3,max_iter=1000)
kmeans.fit(data)
print(kmeans.cluster_centers_)
data["label"]=kmeans.labels_
pd.DataFrame.to_excel(data,r"F:\rock_image\kmeans_.xlsx",index=True)#将原始数据也写进了excle
print("\n完成模型的训练")




