# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:07:19 2019

@author: Rui Kong
"""
'''
参数说明：算法效果较好，但是可能时间较长
状态产生函数：实际问题具体分析
退温函数：指数退温，K一般接近 1
初温：T
Markov链长度L的选取：一般为100~1000
算法停止准则：可以设置终温，用来结束循环
'''
import numpy as np
import time
import matplotlib.pyplot as plt

def caculate_fitness(x,y):
    return 5*np.cos(x*y)+x*y+y**3

#---------------------------初始化参数------------------------------------------------------------
time1=time.time()
D=10      #变量维数
Xmax=5
Xmin=-5     #变量的取值范围
Ymax=5
Ymin=-5
#-------------------------算法参数初始化*------------------------------
L=200             #链长度
K=0.998           #衰减参数
S=0.01                 #步长因子
T=100              #初始温度
Yz=1e-8            #容差
P =0                #metropolis过程中总接收的点
#------------------------------------------------随机选点初值设定--------------------------
PreX=np.random.rand()*(Xmax-Xmin)+Xmin
PreY=np.random.rand()*(Ymax-Ymin)+Ymin
PrexBestX=PreX
PrexBestY=PreY

PreX=np.random.rand()*(Xmax-Xmin)+Xmin
PreY=np.random.rand()*(Ymax-Ymin)+Ymin
BestX=PreX
BestY=PreY
trace=[]

#--------------------------没迭代一次退火一次，直到满足条件为止---------------------------------------
deta=abs(caculate_fitness(BestX,BestY)-caculate_fitness(PrexBestX,PrexBestY))
while(deta>Yz) and (T>0.001):
    T=K*T
    #----------------------在当前温度下迭代------------------------------------
    for i in range(L):
        #-------------------在此点的附近随机选择下一个点--------------------------
        #-------------------边界条件处理--------------------------------------
        n=0
        while n == 0:    
            NextX=PreX + S*(np.random.rand()*(Xmax-Xmin)+Xmin)
            NextY=PreY + S*(np.random.rand()*(Ymax-Ymin)+Ymin)
            if(NextX>=Xmin and Xmax>=NextX and NextY>=Ymin and Ymax>=NextY):
                n=1
        
        #----------------------是否为全局最优解-------------------------------
        if(caculate_fitness(BestX,BestY)>caculate_fitness(NextX,NextY)):
            #--------------------保留上一个最优解-------------------------
            PrexBestX=BestX
            PrexBestY=BestY
            #---------------------此为最优解-----------------
            BestX=NextX
            BestY=NextY
        #-----------------metropolis过程---------------------------
        if(caculate_fitness(PreX,PreY)>caculate_fitness(NextX,NextY)):
            #---------------------接收最新的结果----------------------------
            PreX=NextX
            PreY=NextY
            P+=1
        else:
            changer = -1*(caculate_fitness(NextX,NextY)-caculate_fitness(PreX,PreY))/T
            p1=np.exp(changer)
            #------------------接收较差的解----------------------------------
            if p1 > np.random.rand():
                PreX = NextX
                PreY=NextY
                P+=1
    trace.append(caculate_fitness(BestX,BestY))
    deta=abs(caculate_fitness(BestX,BestY)-caculate_fitness(PrexBestX,PrexBestY))

print("最优参数X结果：",BestX,"最优的Y：",BestY)
print("最优的适应度结果：",caculate_fitness(BestX,BestY))
print("总共耗费时间为：",time.time()-time1)
plt.plot(trace)
    










