#균일확률변수 만들기

import numpy as np
import pandas as pd
np.random.rand(1)

def x(a):
    return np.random.rand(a)

x(10)


# 베르누이 확률변수 모수:p

p=np.random.rand(1)
def Y(p):
     x=np.random.rand(1)
     return np.where(x<p, 1, 0)


def Y(p,num):
    
    
    x=np.random.rand(num)
    
    return np.where(x<p,1,0)


Y(0.5,100)
Y(p=0.5,num=10000).mean()

#새로운, 확률변수
#가질수 있는값 :0,1,2
#20% 50% 30%

np.random.choice(np.arange(1,4),10,True,np.array([2/5,2/5,1/5]))
np.random.radint(0,2,1000)
np.rando.choice(np.arange(1,3),10,True,np.array([2/10,5/10,3/10]))

num=2
def Y(p,num):
    x= np.random.choice(np.arange(0,3),num,True,np.array([2/10,5/10,3/10]))
    
    return np.where(x<1,0,np.where(x<2,1,2) ) #x와 return 앞에 들여쓰기가 일정해야 돌아간다
                


def z(p,num):
    x= np.random.rand(num)
    
    return np.where(x<0.2,0,np.where(x<0.7,1,2))

z(0.5,1000000).mean()

               
Y(0.5,1000000).mean()       
                
p=np.array([0.2,0.5,0.3])
def a(p,num):
    x= np.random.rand(num)
    p_cum=p.cumsum()
    
    
    return np.where(x<p_cum[0],0,np.where(x<p_cum[1],1,2))                

p=np.array([0.2,0.5,0.3])
a(p,10)
#기대값

0*(1/6) + 1 *(2/6) + 2* (2/6) + 3 *(1/6)



sum(np.arange(4) * np.array([1,2,2,1])/6)

np.arange(4)
y=np.array([1,2,2,1])

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x=y) #매개변수 막대그래프
df.plot.bar() # 데이터 프레임 막대그래프
plt.show()
p=0.3
df=pd.DataFrame( {'확률변수' : [1-p,p]})
df.plot.hist()# 데이터 프레임 히스토그램 만들기

sum(np.arange(2) * np.array([1-p,p]))

#넘파이 배열 생성
def com(n):
    data = np.random.rand(n)
    lis=data[5][-1]
    
    mean=np.mean(lis)
    
    return mean

def com(n):
    data = np.random.rand(10).reshape(5,-1)
    
    lis=data.mean(axis=0)
    return lis
    
com(100)

x=np.random(50000)\
    .resjape(-1,5)
    .mean(axis=1)

a=np.random.rand(10).mean()
#히스토그램 그리기
plt.clf()
plt.hist(com(100),bins=100,alpha=0.7,color="blue") #bin 가로축 개수 alpha 색투명도
plt.title("Histo")
plt.xlabel("value")
plt.ylabel("Frequency")
plt.grid(True) #그래프 내부 선 표시 유무
plt.show()



