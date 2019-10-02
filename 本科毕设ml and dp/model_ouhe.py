# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:15:14 2019

@author: PC
"""
# coding: utf-8

import os
import wave
from PIL import ImageFilter,Image
import tensorflow as tf
import pickle
import xlrd
import numpy as np
import random
import time
import math
import cv2
from interval import Interval
#进行图片的识别与分类        
def classify(image_data):
    
    results=[]
# Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("D:/7lei_73/retrained_labels.txt")]
    
# Unpersists graph from file
    with tf.gfile.FastGFile("D:/7lei_73/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0') 
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]   
        for node_id in top_k[0:2]:
            human_string = label_lines[node_id]
            #labelsresult=[2,4,5,8,1,3,7,6]
            score = predictions[0][node_id]
            results.append({'label': human_string, 'score': '{:.4}'.format(score)})
            #results.append({'label': labelsresult[labelsresult.index(human_string)-1], 'score': '{:.2}'.format(score)})
        #print('%s (score = %.5f)' % (human_string, score))
    '''
    with open("result.csv","a",newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("label",'score'))
        writer.writeheader()
        writer.writerow(results[0])
        writer.writerow(results[1])
    '''
    print(results)
    return results
#数据读取
def get_image_result(path,image_name):
 
    #进行单张图片的处理
    images_url=path+"\\"+image_name
    #print(images_url) #执行到这里
    data = tf.gfile.GFile(images_url, 'rb').read()
    
    result1=classify(data)
   
    data=None
    return result1

#如果准备好了数据则进行excle读取
def load_data(file):
    wb=xlrd.open_workbook(file)
    table=wb.sheets()[0]
    rows=table.nrows
    #culums=table.ncols
    ex_data=[]
    label_data=[]
    for row in range(rows):
        data=table.row_values(row)
        ex_data.append(data[:len(data)-1])
        label_data.append(int(data[-1]))
    print(label_data)   
    return ex_data

#判断wav文件并且通过python进行音频的初步处理，返回的结果已经经过了归一处理
def open_wav(filepath):
    listdir=os.listdir(filepath)
    if listdir.endswith(".wav"or".WAV"):
        f = wave.open(filepath+listdir, "rb") 
        #读取格式信息
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        print('声道数:',nchannels)
        print('量化位数:',sampwidth)
        print('采样频率:',framerate)
        print('采样点数:',nframes)
        #读取波形数据
        str_data = f.readframes(nframes)
        f.close()
        #将波形数据转换为数组
        wave_data = np.fromstring(str_data, dtype=np.short)
        wave_data.shape = -1,2
        wave_data = wave_data.T
        #time = np.arange(0, nframes) * (1.0 / framerate)
        max_num=max(wave_data[0])
        wave_da=wave_data[0]/max_num
        return wave_da
    else:
        print("请放置一段音频文件，且最好以wav格式保存*************")
        
#对得到的数据进行切片处理，以0.85为阈值，且每个片段的长度取为3000个点,数据片段储存在一个dict中
def split_data(firstcolumn):
    m=0
    n=0
    order=[]
    for i in range(0,len(firstcolumn)-1):
        if(i<n):
            continue
        else:
            if(abs(firstcolumn[i])>0.85):
                m+=1
                if(i+2760<len(firstcolumn)):
                    order.append(firstcolumn[i-240:i+2760])
                else:
                    order.append(firstcolumn[i-240:len(firstcolumn)])
                n=i+2760                       
    return order
#加载svm模型，并且进行数据的识别,path为svm模型的路径
def open_model(path):
    with open(path,"rb")as f:
        s=f.read()
        model=pickle.loads(s)
        print("model加载成功")
        return model

#进行声音片段的识别，识别结果进行转换，同时输出其对应的识别准确率prob
def predict_result(model,data):
    length=len(data)
    example=random.sample(range(length),1)[0]#传入可能为多个片段，随机取其中的一个进行预测
    print(example)

    predict_label=model.predict([data[example]])#外面的中括号不能少
    print("强度预测值为：",predict_label)
    return predict_label
#两个模型结果进行对比得出结果
    
def compare_two_model(image_result,sound_result,path,name):
    kinds=[]
    scores=[]
    for i in image_result:
        kinds.append(list(i.values())[0])
        scores.append(list(i.values())[1])
    print("---***--*--***--")
    dic=creat_dict()
    scores=list(map(float,scores))
    canshu1=fangfa(sound_result,kinds,scores,dic)
    if (canshu1==3 or canshu1==2):
        result_crop,kinds1,scores1=solu_image(path,name)
        scores1=list(map(float,scores))
        canshu2=fangfa(sound_result,kinds1,scores1,dic)
        if (canshu2==3 or canshu2==2):
            print("两次识别都没有能够匹配，选择性的输出我们的结果：")
            re=kinds[0] if scores[0]>=scores1[0] else kinds1[0]
            print("最终的结果为：*********************************",re)
def fangfa(sound_,image_,scores_,strength):
    #score=list(map(float,scores_))
    if sound_[0] in strength[image_[0]]:
        print("岩石图片与强度能够达到吻合，此次识别正确，正确的种类为：",image_[0])
    elif (sound_[0] in strength[image_[1]]):
        if scores_[0]<0.6 and scores_[1]>0.2:
            print("此时岩石图片识别正确，强度与To2p匹配，输出Top2为：",image_[1])
        else:
            print("虽然声音与Top2匹配，但是图片的TOP1与Top2二者score相差太大")
            return 2
    else:
        print("强度wu图片无法匹配，重新对图片进行识别判断：-----------------------")
        return 3
def creat_dict():
    di={"b_u":Interval(68.6,70.6),"b_y":Interval(56.7,58.7),"b_h":Interval(63.5,65.5),"h_g":Interval(62,64),
        "s_y":Interval(66,68),"d_l":Interval(54.9,56.9),"x_w":Interval(66.4,68.4)}
    return di   
#进行图片处理，去二次识别
def solu_image(path,name):
    #image=Image.open(path+"\\"+name)
    image=cv2.imread(path+"//"+name)
    h,w=image.shape[:2]
    #image_filter=image.filter(ImageFilter.DETAIL)#细节滤波，是细节更加突出
    #image_fil=image.filter(ImageFilter.EDGE_ENHANCE)#边缘增强滤波
    deg=np.arange(45,405,45)#随机旋转一定的角度45的整数倍
    degree=random.choice(deg)
    heightNew=int(w*abs(math.sin(math.radians(degree)))+h*abs(math.cos(math.radians(degree))))
    widthNew=int(h*abs(math.sin(math.radians(degree)))+w*abs(math.cos(math.radians(degree))))
    matRotation=cv2.getRotationMatrix2D((w/2,h/2),degree,1)
    matRotation[0,2] +=(widthNew-w)/2 
    matRotation[1,2] +=(heightNew-h)/2 
    imgRotation=cv2.warpAffine(image,matRotation,(widthNew,heightNew),borderValue=(255,255,255))
    random_num=np.arange(1,10000)
    random.shuffle(random_num)
    j = random.randint(1,10000)
    random.shuffle(random_num)
    
    path1=path+"//"+"crop"+str(random_num[j])+".jpg"
    random_num+=1
    cv2.imwrite(path1,imgRotation)
    print("裁剪完成****-------")
    data = tf.gfile.FastGFile(path1, 'rb').read()
    print("模型二次加载完成")
    result_again=classify(data)
    data=None
    #get_accuracy(result_again,sound_result,path,"crop.jpg") 
    kinds=[]
    scores=[]
    for i in result_again:
        kinds.append(list(i.values())[0])
        scores.append(list(i.values())[1])
    print(result_again)
    
    return result_again,kinds,scores

#进行封装，将实现的过程尽心封装
def app_run(image_path,dir_name,model,data):
    
    
    result1=get_image_result(image_path,dir_name) 
    result2=predict_result(model,data)
    #model=None#下次重新加载声音模型
    compare_two_model(result1,result2,image_path,dir_name)

#返回每个文件的名字
def get_image_name(path):
    dirs=os.listdir(path)
    return dirs
    
    
if __name__ == '__main__':
    image_path=r"D:/7lei_73/ceshi/x_w"
    dirs_=get_image_name(image_path)
    data_piece=load_data(r"C:\Users\PC\Desktop\23\45\xuanwuyanzhu.xlsx")
    #wave_path="11"
    #wave_data=open_wav(wave_path)
    #data_piece=split_data(wave_data)
    model=open_model(r"D:\7lei-10\svm_m_d")
    for dir_name in dirs_:
        #print(image_path,dir_name)
        app_run(image_path,dir_name,model,data_piece)
        time.sleep(1)
    print("success-----------------")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    