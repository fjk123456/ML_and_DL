# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:31:35 2019

@author: PC
"""

import numpy as np
import xlrd
import csv
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score

#读取excle
def readexcle(filename):
    fh=xlrd.open_workbook(filename)
    table=fh.sheets()[0]
    rows=table.nrows
    exdata=[]
    for row in range(rows):          #获取行数
        data=table.row_values(row) #读取每行的数值
        exdata.append(data)
    return exdata

#数据存在csv中进行读取
def readcsv(filename):
    fileobj=open(filename,"r")
    reader=csv.reader(fileobj)
    firstcolumn=[]
    for row in reader: 
        firstcolumn.append(list(map(float,row)))
    print("完成了初步计划")
    return firstcolumn

#分离数据以及label
def splitnumandlab(data):
    culm=len(data[0][:])
    row=len(data)
    yslabel=[]
    ysnumdata=[]
    for j in range(row):
        yslabel.append(data[j][culm-1])
        ysnumdata.append(data[j][0:culm-1])
    return yslabel,ysnumdata
#获取excle原始数据
y_s_data=readexcle(r"F:\gouzao\V3\V3.xlsx")
#获得数据以及label
label,numdata=splitnumandlab(y_s_data)

x_num=np.array(numdata)
y_lab=np.array(label).astype(np.int64)

#对数据进行归一化,sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
scaler=MaxAbsScaler()
#scaler=MinMaxScaler(feature_range=(0,1))
x_num=scaler.fit_transform(x_num)

'''
#pca降维
from sklearn.decomposition import PCA
pca=PCA(n_components=0.90)
pca.fit(re_num)
newX=pca.fit_transform(re_num) 
'''
#划分训练集以及测试集
x_train, x_test, y_train, y_test = train_test_split(x_num, y_lab, train_size=0.8, random_state=5)

#网格搜索寻找最优参数
from sklearn.model_selection import GridSearchCV
#parameters = {'max_features':["auto","sqrt","log2"],"criterion":["gini", "entropy"]}
parameters = {"criterion":["gini", "entropy"]}
dt_tree = tree.DecisionTreeClassifier()
clf = GridSearchCV(dt_tree, parameters,cv=5)
clf.fit(x_train,y_train)
 
print('The parameters of the best model are:------ ')
print(clf.best_params_)
#d_tree=tree.DecisionTreeClassifier(max_features=clf.best_params_["max_features"],criterion=clf.best_params_["criterion"])
d_tree=tree.DecisionTreeClassifier(criterion=clf.best_params_["criterion"])
d_tree.fit(x_train,y_train)
y_predict=d_tree.predict(x_test)

print ('训练集上的正确率为%2f%%'%( d_tree.score(x_train, y_train)*100))
print( '测试集上的正确率为%2f%%'%( d_tree.score(x_test, y_test)*100))

#混淆矩阵
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
confusion=confusion_matrix(y_predict,y_test)#产生混淆矩阵，结果更加清晰明了
classes = list(set(y_test))#得到label的个数
classes.sort()#将label按照顺序从小到大进行排列
#绘图，显示混淆矩阵
plt.imshow(confusion, cmap=plt.cm.Reds)
indices = range(len(confusion))
#设置x,y轴上面的数字
plt.xticks(indices, classes)
plt.yticks(indices, classes)
#颜色bar,对应不同的深度
plt.colorbar()#单个试图，多子图fig.colorbar()
#定义坐标轴名称
plt.xlabel("预测值",fontproperties=font)
plt.ylabel("实际值",fontproperties=font)

for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
         plt.text(first_index, second_index, confusion[first_index][second_index])

plt.legend()
plt.show()

#将预测值与实际label写入文件中进行对比
filename = r'F:\岩石基因图谱\二次重做\单斜辉石\dt\y_test.csv'
with open(filename ,'w')as f:
    for number in y_test:
        f.write(str(number)+'\n')
        
#将#将rbf_svr_y_predict写入文件
filename__ = r'F:\岩石基因图谱\二次重做\两类矿物合并\dt\y_predict.csv'
with open(filename__,'w')as f:
    for number in y_predict:
        f.write(str(number)+'\n')
        
importances=d_tree.feature_importances_
with open(r'F:\岩石基因图谱\二次重做\两类矿物合并\dt\importanc.csv',"w") as file:
    for i in importances:
        file.write(str(i)+"\n")

import pickle
fw = open(r"F:\岩石基因图谱\二次重做\两类矿物合并\dt\dt_",'wb')
pickle.dump(d_tree,fw)
fw.close()













