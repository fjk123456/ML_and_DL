# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:49:25 2019

@author: Rui Kong
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt

'''
差分进化算法：初始化、变异、交叉、边界处理、计算fitness、选择
种群数量一般为4D-10D，必须大于4
变异算子F一般取为0.5
交叉算子CR一般为0.1或0.9
最大进化代数G为100-500
此处为求最小值的问题，若要进行修改，则需要将fitness的地方全部进行修改。
'''
def caculate_fitness(x):
    return sum(x**2)
population_num=100              #种群数量
variable_dimension=10           #变量的个数
iteration_count=200
FC=0.5
CROSS_OVER=0.1
Xlarge=20
Xsmall=-20
Yz=1e-6     #阈值
#初始化-----------------------
initial_population=np.zeros((variable_dimension,population_num))

cross_population=np.zeros((variable_dimension,population_num))

selection_population=np.zeros((variable_dimension,population_num))
#赋予初始值
initial_population = np.random.rand(variable_dimension,population_num)*(Xlarge-Xsmall) + Xsmall

fitness1=[]       #存放每次计算的fitness
for i in range (population_num):
    fitness1.append(caculate_fitness(initial_population[:,i]))
print(min(fitness1))
trace=[] #记录每次迭代过程中的最优值
trace.append(min(fitness1))
#---------------------差分进化选择--------------------------------------
for i in range(population_num):
    #------------------------自适应变异算子-------------------------
    lamda = np.exp(1-iteration_count/(iteration_count+1-i))
    F = FC*2**lamda
    #------r1\r2\r3与m互不相同
    for j in range(population_num):
        r1 = np.random.randint(0,population_num)
        while r1 == j :
            r1 = np.random.randint(0,population_num)
        
        r2 = np.random.randint(0,population_num)
        while r2==j or r2==r1:
            r2 = np.random.randint(0,population_num)
        
        r3 = np.random.randint(0,population_num)
        while r3==j or r3==r2 or r3==r1:
            r3 = np.random.randint(0,population_num)
            
        cross_population[:,j] = initial_population[:,r1] + F*(initial_population[:,r2]-initial_population[:,r3])
          
        #-------------------交叉操作cross_over-------------------------------------------------------------
        r=np.random.randint(0,variable_dimension)
        for n in range(variable_dimension):
            cross_posi=np.random.rand()
            if cross_posi<=CROSS_OVER or r==n:
                selection_population[n,:] = cross_population[n,:]
            else:
                selection_population[n,:] = initial_population[n,:]
        
        #--------------------边界处理----------------------------------------
        for n in range(variable_dimension):
            for m in range(population_num):
                if(selection_population[n,m] < Xsmall) or (selection_population[n,m]>Xlarge):
                    selection_population[n,m] = np.random.rand()*(Xlarge-Xsmall) + Xsmall
                    
        #--------------------选择操作-----------------------------------------------------------
        fitness2=[]
        for n in range(population_num):
            fitness2.append(caculate_fitness(selection_population[:,n]))
        for m in range(population_num):
            if(fitness2[m]<fitness1[m]):                         #调整最大最小值的位置，可以改变不同的问题
                initial_population[:,m] = selection_population[:,m]
        
        for m in range(population_num):
            fitness1[m]=caculate_fitness(initial_population[:,m])
        trace.append(min(fitness1))                               #修改的地方
        if min(trace)< Yz :
            break
#-------------------------------------------------------end----------------------------
index=np.argsort(fitness1)
BestX=initial_population[:,index[0]]
BestY=min(fitness1)
print("最优值与最优结果分别为：",BestX,"-------------",BestY)
plt.plot(trace)

                
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        