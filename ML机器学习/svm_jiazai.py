# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:01:18 2019

@author: PC
"""


import numpy as np
import xlrd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics


#从excle中获取数据
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
        #ysnumdata.append(data[j][0:culm-1])#获取全部行作为一个样本
        ysnumdata.append(data[j][265:290])#采用[1500:2201]的数据进行训练
    return yslabel,ysnumdata
#获取excle原始数据
y_s_data=readexcle(r"C:\Users\lenovo\Desktop\fjk1\10类.xlsx")
#获得数据以及label
label,numdata=splitnumandlab(y_s_data)
#利用np模块转为array以及获取训练集以及测试集
x_num=np.array(numdata)
y_lab=np.array(label).astype(np.int64)
#对数据进行归一化,sklearn.preprocessing
'''
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
re_num=scaler.fit_transform(x_num)
'''
#划分训练集以及测试集
x_train, x_test, y_train, y_test = train_test_split(x_num, y_lab, train_size=0.8, random_state=1)

from sklearn.model_selection import GridSearchCV
cnum=np.linspace(1,2,endpoint=False)
gnum=np.linspace(0.01,0.1,10,endpoint=False)
parameters = {'C':cnum,"gamma":gnum}
rbf_svc = svm.SVC()
clf = GridSearchCV(rbf_svc, parameters,cv=5,n_jobs=1)
clf.fit(x_train,y_train)  
print('The parameters of the best model are:------ ')
print(clf.best_params_)

#设置相应的参数
#model=svm.SVC(kernel="linear",C=0.95)  #线性核函数
#model=svm.SVC(kernel="rbf",C=clf.best_params_["C"],gamma=clf.best_params_['gamma'])   #高斯核函数
model=svm.SVC(kernel="rbf",C=clf.best_params_["C"],gamma=clf.best_params_['gamma'],
              probability=True)#可以输出相应的概率
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

print ('训练集上的正确率为%2f%%'%( model.score(x_train, y_train)*100))
print( '测试集上的正确率为%2f%%'%( model.score(x_test, y_test)*100))
'''
print ("RMSE:------",np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
print("MAPE:-------",np.mean(np.abs((y_test - y_predict) / y_test)) * 100)
print("MAE:--------",metrics.mean_absolute_error(y_test, y_predict))
'''
#将预测值与实际label写入文件中进行对比
filename = 'y_test.csv'
with open(filename ,'w')as f:
    for number in y_test:
        f.write(str(number)+'\n')
        
#将#将rbf_svr_y_predict写入文件
filename__ = 'y_predict.csv'
with open(filename__,'w')as f:
    for number in y_predict:
        f.write(str(number)+'\n')
               
#产生混淆矩阵，判断svm分类发具体情况，算出各项指标
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
recall=recall_score(y_test,y_predict,average=None)
print("recall的准确率为---------",recall)
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
        
#保存模型
import pickle
with open("svm_m_d","wb") as f:
    pickle.dump(model,f)

#加载模型的方法
'''
from keras.externals import joblib
model=joblib.dump("svm.pkl")
'''  