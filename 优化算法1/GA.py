# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:26:25 2019

@author: Rui Kong
"""

#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import math
import numpy as np

#计算适应度方法
def calculate_fitness(x):
    y=x+10*math.sin(5*x)+7*math.cos(4*x)
    return y

#轮盘赌法，对数据fitness进行取舍
def selection(population,fvalue):
    fitness_sum=np.add.accumulate(fvalue)   #依次相加得到求和的结果，最后返回
    fitness_sum/=sum(fvalue)                   #依次求每个所占的比例，轮盘赌法
    #print(fitness_sum)
    new_population=[]
    
    for i in range(len(population)):
        rand=np.random.uniform(0,1)
        for j in range(len(value)):
            if j==0:
                if rand <= fitness_sum[j]:
                    new_population.append(population[j])
                    break
            else:
                if fitness_sum[j-1]<rand and rand<=fitness_sum[j]:
                    new_population.append(population[j])
                    break
    print(str(len(new_population))+"selection")
    return new_population
    
#交叉操作，生成新的个体
def crossover(population,PC):
    half_num=int(len(population)/2)
    print(half_num)
    np.random.shuffle(population)
    parent_1=population[:half_num]
    parent_2=population[half_num:]
    print(str(len(parent_1)))
    print(str(len(parent_1)))
    new_cross_population=[]
    for i in range(half_num):
         rand_cross = np.random.uniform(0,1)
         if rand_cross > PC:
             location= np.random.randint(1,len(population[0])-1)
             son1=parent_1[i][:location]+parent_2[i][location:]
             son2=parent_2[i][:location]+parent_1[i][location:]       #前后两部分进行替换，实现crossover
             new_cross_population.append(son1)
             new_cross_population.append(son2)
         else:
             new_cross_population.append(parent_1[i])
             new_cross_population.append(parent_2[i])
    print(str(len(new_cross_population))+"cross")
    return new_cross_population

#变异操作，加强多样性,随机单点变异
def mutation(population,PM):
    length_single=len(population[0])
    
    for i in range(len(population)):
        rand_mutation = np.random.uniform(0,1)
        if rand_mutation<=PM:
            mutation_position=np.random.randint(0,length_single)
            if mutation_position > 0 and mutation_position < length_single-1 :
                if population[i][mutation_position]=='1':
                    population[i]=population[i][:mutation_position]+'0'+ population[i][mutation_position+1:]
                else:
                    population[i]=population[i][:mutation_position]+'1'+ population[i][mutation_position+1:]
            elif mutation_position == length_single-1:
                if population[i][mutation_position]=='1':
                    population[i]=population[i][:mutation_position]+'0'
                else:
                    population[i]=population[i][:mutation_position]+'1'
            else:
                if population[i][mutation_position]=='1':
                    population[i]='0'+population[i][1:]
                else:
                    population[i]='1'+population[i][1:]
            
    print(str(len(population))+"mutation")      
    return population

#二进制转为10进制,范围为（a,b)
def decode(x):
    y= 0+10*int(x,2)/(2**len(x)-1)
    return y

#
def fitness(population):
    fitness=[]
    for i in range (len(population)):
        fitness.append(calculate_fitness(decode(population[i])))
        if(fitness[i]<0):
            fitness[i]=0.0
    return fitness

             
        

PC=0.8                    #交叉率
PM=0.05                    #变异率
GEN_SIZE=100             #种群总数
SINGLE_SIZE=10           #单个染色体的长度
ITERATION_SIZE=100      #总的迭代次数

#初始化所有population
population=[]

for i in range(GEN_SIZE):
    single=""
    for j in range(SINGLE_SIZE):
         single+=str(np.random.randint(0,2))
    population.append(single)
    
fitness_value=[]
for i in range(ITERATION_SIZE):
    value=fitness(population)
    population_new=selection(population,value)
    #crossover
    offspring =crossover(population_new,0.8)
    #mutation
    population=mutation(offspring,0.1)
    result=[]
    for j in range(len(population)):
        result.append(calculate_fitness(decode(population[j])))
    fitness_value.append(max(result))              #每次迭代的最大值
plt.plot(fitness_value)
print("最大的仿真结果为：",max(fitness_value))
plt.axhline(max(fitness_value), linewidth=1, color='r')    
     