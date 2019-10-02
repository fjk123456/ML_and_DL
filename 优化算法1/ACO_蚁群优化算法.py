# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:26:55 2019

@author: Rui Kong
"""
import numpy as np
import matplotlib.pyplot as plt
'''
参数说明：
信息素启发因子α一般取为[1,4]
期望启发因子β一般取为[3,5]，与α合理搭配
信息素蒸发系数ρ：ρ越小，表示之前路径再次被选择的可能性较大，合理选择
蚂蚁群数量m一般取为10~50
信息强度Q对算法性能的影响：不做特别的考虑，可以任意选择
最大进化代数G：100~500
'''
def caculate_fitness(x,y):
    return 20*(x**2-y**2)**2-(1-y)**2-3*(1+y)**2+0.3
#-------------------------------------参数初始化---------------------------------------
m=20
dimension=2
G=200
Rh0=0.9
P0=0.2
Xmax=5
Xmin=-5
Ymax=5
Ymin=-5
X_single = np.zeros((m,dimension))
Tau=np.zeros(m)
P=np.zeros((G,m))
Trace=[]
#-----------------------------------------------------------------------------------
for i in range(m):
    X_single[i,0] = np.random.rand()*(Xmax-Xmin) + Xmin
    X_single[i,1] = np.random.rand()*(Ymax-Ymin) + Ymin
    Tau[i]= caculate_fitness(X_single[i,0],X_single[i,1])    
step = 0.1
for j in range(G):
    lamda = 1/G
    index1=np.argsort(Tau)
    Tau_best=Tau[index1[0]]
    #--------------------------------------计算状态转移概率---------------------------------------
    for i in range(m):
        P[j,i]=(Tau[index1[0]]-Tau[i])/Tau[index1[0]]
    #------------------------------------------位置更新---------------------------------------------
    for i in range(m):
        #-------------------------------------局部搜索--------------------------------------
        if P[j,i]<P0:
            temp1 = X_single[i,0]+(2*np.random.rand()-1)*step*lamda
            temp2 = X_single[i,1]+(2*np.random.rand()-1)*step*lamda
        else:
            temp1 = X_single[i,0]+(np.random.rand()-0.5)*(Xmax-Xmin)
            temp2 = X_single[i,1]+(np.random.rand()-0.5)*(Ymax-Ymin)
        
        if temp1<Xmin:
            temp1=Xmin
        if temp1>Xmax:
            temp1=Xmax
        if temp2 < Ymin:
            temp2=Ymin
        if temp2>Ymax:
            temp2=Ymax
        if(caculate_fitness(temp1,temp2)<caculate_fitness(X_single[i,0],X_single[i,1])):
            X_single[i,0]=temp1
            X_single[i,1]=temp2
        #------------------------------------更新信息素----------------------------------------
        for i in range(m):
            Tau[i]=(1-Rh0)*Tau[i]+caculate_fitness(X_single[i,0],X_single[i,1])
        index2=np.argsort(Tau)
        Trace.append(caculate_fitness(X_single[index2[0],0],X_single[index2[0],1]))
index3=np.argsort(Tau)
minX=X_single[index3[0],0]
minY=X_single[index3[0],1]
minValue=caculate_fitness(minX,minY)
print("最优参数分别为：",minX,"最优的Y值为：",minY,"最优的输出结果为：",minValue)
plt.plot(Trace)

    
 









