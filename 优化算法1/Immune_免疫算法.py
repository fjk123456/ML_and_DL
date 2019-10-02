# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:00:24 2019

@author: Rui Kong
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
'''
免疫优化算法：全局寻优，主要参数介绍
抗体种群大小NP，一般取为10-100，不适宜超过200
免疫选择比例一般为NP大小的10%-50%
抗议克隆扩增的倍数N一般为5-10
种群刷新比例一般不超过种群NP的50%
最大进化代数G一般多取为100-500

优化函数作为实例：F(X) = X1**2+X2**2+....+X10**2     
'''
anti_len =10                     #抗体维度
anti_population=100              #抗体个数
Xlarge=20
Xsmall=-20
Iteration_num=300                 #迭代次数
Pm=0.8                             #变异概率,越大收敛越快
alfa=1                           #激力度系数
belta=1
detail=0.2                          #相似度阈值
anti_gen=0                           #免疫代数
clone_num=20                       #克隆个数
deta0 =1*Xsmall                     #领域范围初始值

#----------------------- 参数说明------------------------------------------------
def caculate_fitness(x):
    return sum(x**2)
#-----------------------------------计算fitness--------------------------------
population=np.random.rand(anti_len,anti_population)*(Xlarge-Xsmall) + Xsmall
MSSL_fitness=[]
for i in range(anti_population):
    MSSL_fitness.append(caculate_fitness(population[:,i]))

#------------------------------------------------计算个体浓度与激励度--------------------------------------
ND=[]          #相似度数组
for i in range(anti_population):
    nd=[]
    for j in range(anti_population):
       
        nd.append(sum(np.sqrt((population[:,i]-population[:,j])**2)))
        if(nd[j]<detail):
            nd[j]=1
        else:
            nd[j]=0
    ND.append(sum(nd)/anti_population)
    
MSSL_fitness = alfa*np.array(MSSL_fitness) -belta* np.array(ND)

#---------------------------------------------按照激励度系数进行升序排列--------------------
index=np.argsort(MSSL_fitness)
sortpopulation=[]
for i in range(anti_population):
    sortpopulation.append(population[:,index[i]])

#-----------------------------------------免疫循环------------------------------------------
af=[]
bf=[]
aND=[]
bND=[]

trace=[]
while anti_gen < Iteration_num:
    for i in range(int(anti_population/2)):
        #------------------选择激励度前anti-population/2进行免疫操作---------------------------------------
        aMSSL=[]
        bMSSL=[]
        bND=[]
        singele=sortpopulation[i]
        #############################################################################################################
        clone=[]
        for i in range(clone_num):
            clone.append(singele)
        clone=np.array(clone).T
       
        deta =deta0/(anti_gen+1)
        for j in range(clone_num):
            for n in range(anti_len):
                #-----------------变异--------------------------------------------
                rand = np.random.rand()
                if rand< Pm:
                    clone[n,j] = clone[n,j] + (rand-0.5)*deta
                    #print(clone[n,j])
                if (clone[n,j] > Xlarge) or (clone[n,j] < Xsmall):
                    clone[n,j] = np.random.rand()*(Xlarge-Xsmall) + Xsmall
        clone[:,i] = sortpopulation[i]                     #-----------------保留克隆的源个体--------------
        #---------------------------------克隆抑制，保留亲和度最高的个体-------------------------------
        clone_MSSL=[]
        for n in range(clone_num):
            clone_MSSL.append(caculate_fitness(clone[:,n]))
        index_1=np.argsort(clone_MSSL)
       
        aMSSL.append(clone_MSSL[index_1[0]])
        clone_sort=[]
        for j in range(len(index_1)):
            clone_sort.append(clone[:,index_1[j]])
        af.append(clone_sort[0])
        
        
        
    #-----------------------免疫种群激励制度-------------------------------------
    for nm in range(int(anti_population/2)):
        nda=[]
        for j in range(int(anti_population/2)):
            
            nda.append(sum(np.sqrt((af[nm]-af[j])**2)))
            
            if nda[j] < detail:
                nda[j]=1
            else:
                nda[j]=0
        aND.append(sum(nda)/anti_population/2)
    
    aMSSL = alfa*np.array(aMSSL) *(Xlarge- Xsmall) + Xsmall
    
    #-------------------------------种群刷新------------------------------------
    bf=np.random.rand(anti_len,int(anti_population/2)) *(Xlarge-Xsmall) +Xsmall
    for i in range(int(anti_population/2)):
        bMSSL.append(caculate_fitness(bf[:,i]))
        
    #-------------------------新生成种群激励制度-----------------------------------
    for nm in range(int(anti_population/2)):
        ndc=[]
        for j in range(int(anti_population/2)):
            ndc.append(sum(np.sqrt((bf[:,nm] - bf[:,j])**2)))
            if ndc[j] < detail:
                ndc[j]=1
            else:
                ndc[j]=0
        bND.append( sum(ndc)/anti_population/2)
    bMSSL = alfa*np.array(bMSSL) - belta*bND
    
    #---------------------------------免疫种群与新生种群合并-------------------------------------------------
    #print((np.array(af).T).shape)
    f1=np.concatenate((np.array(af).T,bf),axis=1)
    MSSL1=np.concatenate((aMSSL,bMSSL))
    index_3=np.argsort(MSSL1)
    #print(index_3)
    for m in range (len(index_3)):
        
        sortpopulation[m] = f1[:,m]
    anti_gen=anti_gen+1
    af=[]
    print(caculate_fitness(sortpopulation[0]))
    trace.append(caculate_fitness(sortpopulation[0]))

#------------------------------输出最优结果----------------------------------------------------
BestX=sortpopulation[0]
BestY=min(trace)
print("最优的参数为：",BestX,"最优的输出结果为：",BestY)
plt.plot(trace)


































        
            