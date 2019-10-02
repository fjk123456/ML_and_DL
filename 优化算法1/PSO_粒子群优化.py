# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:17:20 2019

@author: Rui Kong
"""
'''
参数说明：PSO-粒子群优化算法
粒子群规模N一般为100-200
惯性权重W一般多为[0.8,1.2]
加速常数c1与c2,一般二者取值相等
粒子最大的速度vmax、vmin及变量的范围、
边界处理，当边界不满足范围，重新随机产生一个新的数
'''
import numpy as np
import matplotlib.pyplot as plt

def caculate_fitness(x):
    return sum(x**2)

#--------------------------------------参数初始化--------------------------------------
N=200     #粒子个数
D=10      #维度
T=500     #迭代次数
c1=1.5
c2=1.5      #加数参数
W=0.8     #惯性权重
Xmax=20
Xmin=-20    #定位变量的范围
Vmax=5
Vmin=-5     #定义粒子移动速度

#-----------------------------------------位置与初始速度的初始化------------------------------------------
x=np.random.rand(N,D)*(Xmax-Xmin)+Xmin
v=np.random.rand(N,D)*(Vmax-Vmin)+Vmin
p=x
pbest=np.ones(N)        #储存每一代中每一个的最优值
g=np.ones((1,D))
gbest=np.inf
global_best=np.ones(N)
#------------------------------------计算初始代的最优值及位置-----------------------------------------------------------
for i in range(N):
    pbest[i]=caculate_fitness(x[i,:])

#--------------------------全局最优与最佳位置-------------------------------------------------------
g=np.ones((1,D))               #最优个体储存
gbest=np.inf                      #第一轮最优值
for i in range(N):
    if(pbest[i]<gbest):
        g=p[i,:]
        gbest=pbest[i]
gb=np.ones((1,T))                #储存每轮中的最优值

#----------------------------进行迭代，直到满足条件或者达到迭代次数-------------------------------------
for i in range(T):
    for j in range(N):
        #-*--------------------更新粒子最优值与位置-----------------------
        if(caculate_fitness(x[j,:]) < pbest[j]):
            p[j,:] = x[j,:]
            pbest[j] = caculate_fitness(x[j,:])
        #--------------------更新全局最优位置与最优值----------------------------------------
        #print(gbest)
        if(pbest[j]<gbest):
            g=p[j,:]
            gbest = pbest[j]
        #------------------更新速度与位置值-----------------------------------------
        
        v[j,:] = W*v[j,:]+c1*np.random.rand()*(p[j,:]-x[j,:])+c2*np.random.rand()*(g-x[j,:])
        #print(v[j,:])
        x[j,:] = x[j,:] + v[j,:]
        
        #---------------------------------边界条件处理----------------------------------
        for n in range(D):
            if(v[j,n]>Vmax) or (v[j,n]<Vmin):
                v[j,n] = np.random.rand()*(Vmax-Vmin)+Vmin
            
            if(x[j,n]>Xmax) or (x[j,n]<Xmin):
                x=np.random.rand(N,D)*(Xmax-Xmin)+Xmin
        
        gb[0,i]=gbest
        
print("最优的取值为：",g)
print("最优的优化结果为：",min(gb[0]))
plt.plot(gb[0])































