# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:58:33 2019

@author: Rui Kong
"""

import math
import numpy as np
import matplotlib.pyplot as plt

#计算适应度值
def caculate_fitness(x):
    return sum(x**2)
#对每一个进行norm归一化处理，相当于matlab中的norm函数
def normlization(x):
    array1=np.array(x).reshape(1,-1)
    print(array1)
    array2=[x**2 for x in array1]
    print(array2)
    array2_sum=sum(array2)
    norm=math.sqrt(array2_sum[0])
    print(norm,"norm")
    return norm


eta=0.98
c=5               #步长与两须之间的关系
step=5         # 初始步长
iteration_num=300           #迭代次数
variable_num=2             #优化参数的个数
beetle=np.random.rand(variable_num,1)
best_beetle=beetle
best_fitness=caculate_fitness(beetle)

best_fitness_array=[]                #储存每次迭代的最优的结果
best_fitness_array.append(best_fitness)  
#循环迭代进行调整
for i in range(iteration_num):
    distance=step/c            #左右两须之间的距离
    direction_1=np.random.rand(variable_num,1)
    #随机方向
    direction=direction_1/np.linalg.norm(direction_1)
    #更新左右两边的虫须
    left_beard = beetle+direction*distance/2
    left_fitness = caculate_fitness(left_beard)
    #print(left_fitness,"left")
    
    right_beard = beetle - direction*distance/2
    right_fitness = caculate_fitness(right_beard)
    #print(right_fitness,"right")
    #更新甲壳虫的位置
    beetle=beetle - step*direction*np.sign(left_fitness-right_fitness)              #建号表示往小的方向移动
    print(beetle)
    fitness_change=caculate_fitness(beetle)
    
    best_fitness_array.append(fitness_change)
    if fitness_change < best_fitness:
        best_fitness=fitness_change
        best_beetle=beetle
    '''
    if fitness_change > 0.95:           #准确率大于95%则可以退出循环
        best_fitness=fitness_change
        best_beetle=beetle
        break
    '''
    step = eta*step
print("最佳的个体为：",best_beetle)
print("最佳的表现结果为：",min(best_fitness_array))
plt.plot(best_fitness_array)  

                