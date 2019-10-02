# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:09:52 2019

@author: PC
"""

import pickle
import xlrd
import os

#打开模型，利用pickle模块
def open_model(path):
    with open(path,"rb")as f:
        s=f.read()
        model=pickle.loads(s)
        return model
#加载数据
def load_data(file):
    wb=xlrd.open_workbook(file)
    table=wb.sheets()[0]
    rows=table.nrows
    #culums=table.ncols
    ex_data=[]
    label_data1=[]
    label_data2=[]
    for row in range(rows):
        data=table.row_values(row)
        ex_data.append(data[:len(data)-2])
        label_data1.append(int(data[-2]))
        label_data2.append(int(data[-1]))
    print(label_data2)   
    return ex_data,label_data1,label_data2

#进行预测
def predict_result(model,data):
    length=len(data)
    predict_label=[]
    #prob=[]
    for i in range(length):
        result=model.predict([data[i]])#svm模型进行预测的时候，必须加一个[]
        m=result[0]
        #prob.append(model.predict_proba(data[i]))
        predict_label.append(m)
    print(predict_label)
    
    return predict_label

#计算准确率
def score_acc(name,c_s_label,predict_label1):
    length=len(c_s_label)
    #print(length)
    #print(predict_label1)
    a=0
    for i in range(length):
        if(c_s_label[i]==predict_label1[i]):
            a+=1
        else:
            print(name+"音频的第%d个片段预测错误"%(i+1))
    accuracy=a/length
    return accuracy
#进行文件夹的寻找
def file_dir(path):
    files=os.listdir(path)
    return files

#用于条件判断到底执行乃个svm模型
def chose_model(path,data,predict_1):
    results=[]
    for i in range(len(predict_1)):
        
        if predict_1[i] ==1:
            model1=open_model(path[0])
            result=model1.predict([data[i]])
            m=result[0]
            results.append(m)
        elif predict_1[i] ==2:
            model2=open_model(path[1])
            result=model2.predict([data[i]])
            m=result[0]
            results.append(m)
        else:
            model3=open_model(path[2])
            result=model3.predict([data[i]])
            m=result[0]
            results.append(m)
    return results
            
            

   
if __name__=="__main__":
    model_path=r"C:\Users\PC\Desktop\三模型图片\svm三类93.5065\svm_1"
    path_list=[]
    model_path1=r"C:\Users\PC\Desktop\三模型图片\123\svm_沉积岩\svm_chenjiyan"
    model_path2=r"C:\Users\PC\Desktop\三模型图片\123\svm_火成岩\svm_huochengyan"
    model_path3=r"C:\Users\PC\Desktop\三模型图片\123\svm_变质岩\svm_bianzhi"
    path_list.append(model_path1)
    path_list.append(model_path2)
    path_list.append(model_path3)
    dir_path=r"C:\Users\PC\Desktop\三模型图片\v3txt\123"
    model=open_model(model_path)
    files=file_dir(dir_path)
    num_file=len(files)
    for i in range(num_file):
        file_split=files[i].split(".")
        initial_data,label1,label2=load_data(dir_path+os.sep+files[i])
        predict_label1=predict_result(model,initial_data)
        #print(predict_label1)
        jieguo_data=chose_model(path_list,initial_data,predict_label1)
        accura=score_acc(file_split[0],label2,jieguo_data)
        print("最终-----{}------的测试结果准确率为{}---------------------".format(file_split[0],accura))
    print("完成所有的测试准确率测试")