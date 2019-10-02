# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:01:18 2019

@author: PC
"""


import numpy as np
import xlrd
import csv
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

#数据存在csv中进行读取
def readcsv(filename):
    fileobj=open(filename,"r")
    reader=csv.reader(fileobj)
    firstcolumn=[]
    for row in reader: 
        firstcolumn.append(list(map(float,row)))
    print("完成了初步计划")
    return firstcolumn

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

#分离数据以及label
def splitnumandlab(data):
    culm=len(data[0][:])
    row=len(data)

    yslabel=[]
    ysnumdata=[]
    for j in range(row):
        yslabel.append(data[j][culm-1])
        ysnumdata.append(data[j][0:culm-2])
    return yslabel,ysnumdata
#获取excle原始数据
y_s_data=readexcle(r"F:\岩石基因图谱\二次重做\两类矿物合并\两类合并.xlsx")
#y_s_data=readcsv(r"C:\Users\PC\.spyder-py3\瞎搞的python\mfcc.csv")
#读取csv文件
#y_s_data=readcsv(r"C:\Users\PC\Desktop\456\zong.csv")
#获得数据以及label
label,numdata=splitnumandlab(y_s_data)
#主要用于pca降维，使得数据计算更加便捷快速
'''
from sklearn.decomposition import PCA
pca=PCA(n_components=0.8)
pca.fit(numdata)
newX=pca.fit_transform(numdata) 
'''
#利用np模块转为array以及获取训练集以及测试集
x_num=np.array(numdata)
y_lab=np.array(label).astype(np.int64)

#对数据进行归一化,sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
#scaler=MaxAbsScaler()#归一到[-1，1],考虑到大量0的存在
scaler=MinMaxScaler(feature_range=(0,1))
re_num=scaler.fit_transform(x_num)

#划分训练集以及测试集
x_train, x_test, y_train, y_test = train_test_split(re_num, y_lab, train_size=0.8, random_state=5)

from sklearn.model_selection import GridSearchCV
k=np.arange(2,6)
weight=['uniform','distance']
p_num=np.arange(1,6)
parameters = [{'n_neighbors':k,"weights":['uniform'],"p":[2]},{'n_neighbors':k,"weights":['distance'],"p":p_num}]
rbf_knn = KNeighborsClassifier()
clf = GridSearchCV(rbf_knn, parameters,cv=5,n_jobs=1)
clf.fit(x_train,y_train)  
print('The parameters of the best model are:------ ')
print(clf.best_params_)

#model=svm.SVC(kernel="linear",C=0.95)  #线性核函数,不包含K折交叉验证
model=KNeighborsClassifier(n_neighbors=clf.best_params_["n_neighbors"],weights=clf.best_params_['weights'],p=clf.best_params_["p"])  
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

print ('训练集上的正确率为%2f%%'%( model.score(x_train, y_train)*100))
print( '测试集上的正确率为%2f%%'%( model.score(x_test, y_test)*100))

#多用于分类进行观测


'''
print ("RMSE:------",np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
print("MAPE:-------",np.mean(np.abs((y_test - y_predict) / y_test)) * 100)
print("MAE:--------",metrics.mean_absolute_error(y_test, y_predict))
'''
#将预测值与实际label写入文件中进行对比
filename = r'F:\岩石基因图谱\二次重做\两类矿物合并\knn\y_test.csv'
with open(filename ,'w')as f:
    for number in y_test:
        f.write(str(number)+'\n')
        
#将#将rbf_svr_y_predict写入文件
filename__ = r'F:\岩石基因图谱\二次重做\两类矿物合并\knn\y_predict.csv'
with open(filename__,'w')as f:
    for number in y_predict:
        f.write(str(number)+'\n')
               
#产生混淆矩阵，判断svm分类发具体情况，算出各项指标
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

#print("数据集上面的总的精度为%2f"%(accuracy_score(y_test,y_predict)))
#print("数据集上面的总的精度为%2f"%(precision_score(y_test,y_predict,average=None)))
#print("数据集上面的召回率为%2f"%(recall_score(y_test,y_predict,average=None)))#默认average情况下，只适用二分类，None模式适合多分类
        
#保存模型
import pickle
with open(r"F:\岩石基因图谱\二次重做\两类矿物合并\knn\knn_","wb") as f:
    pickle.dump(model,f)

'''
#加载模型的方法

from keras.externals import joblib
model=joblib.dump("svm.pkl")
'''  