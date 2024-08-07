# 표본평균에 따른 랜덤값 뽑기 + 신뢰구간 +표본분산 함수로 구해보기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(20240729)


old_seat=np.arange(1,29)
new_seat=np.random.choice(old_seat,28, replace=False)

result=pd.DataFrame(
    {"old_seat" : old_seat,
    "new_seat"  : new_seat}
)


result.to_csv("result.csv")

k=np.linspace(0,8,8)
y= 2*k**2
plt.scatter(k,y,s=3)
#plt.plot(k,y,color="black")
plt.show()
plt.clf()


x=np.linspace(-8,8,100)
y=x**2
plt.scatter(x,y,s=5)
plt.plot(x,y,color="black")

# xlim과 ylim과 같이 사용 x
#plt.axis("equal") # 그래프 비율을 실제 비율에 맞게 바꿔는것

#x,y축 범위 설정
plt.xlim(-10,10)
plt.ylim(0,40)
plt.gca().set_aspect("equal",adjustable="box")
plt.show()
plt.clf()

from scipy.stats import norm

s=np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
s.mean()
len(s)

z_005=norm.ppf(0.95,loc=0,scale=1)
z_005
#신뢰구간 https://bioinformaticsandme.tistory.com/256
x.mean() +z_005 *6 /np.sqrt(16) 
x.mean() -z_005
plt.plot(s,ss,color="black")
plt.show()
plt.clf()

np.random.seed(20240729)
x=norm.rvs(loc=3,scale=5,size=10000)
x


np.mean(x**2)

for i in x:
    ab=x-3**2

ab
ss=1/(len(x)-1)*sum(ab)**2


x

x_bar=x.mean()
s_2= sum((x-x_bar)**2)  / (10000-1) #x_bar를 사용한 이유는 기대값을 모를때 이와 가장 유사한 평균사용 몬테카를로 적분
s_2

np.var(x,ddof=1) #n-1로 나눈 값(표본분산)

np.var(x,ddof=0) #n로 나눈 값 표본분산아님 np.var(x)의 디폴트값



x=norm.rvs(loc=3,scale=5,size=20)
np.var(x)
np.var(x,ddof=1)


# HW5 
#균일분포 (3, 7)에서 20개의 표본을 뽑아서 분산을 2가지 방법으로 추정해보세요.


#1.n-1로 나눈 것을 s_2, n으로 나눈 것을 k_2로 정의하고, s_2의 분포와 k_2의 분포를 그려주세요!  (10000개 사용)

#1.n-1로 나눈 것을 s_2, n으로 나눈 것을 k_2로 정의하고, s_2의 분포와 k_2의 분포를 그려주세요! (10000개 사용)

#2.각 분포 그래프에 모분산의 위치에 녹색 막대를 그려주세요.
#3.결과를 살펴보고, 왜 n-1로 나눈 것을 분산을 추정하는 지표로 사용하는 것이 타당한지 써주세요!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
x=uniform.rvs(loc=3,scale=4,size=10000*20) #균일분포 (3, 7)에서 20개의 표본 scale은 거리 그래서 3,7일때 7-3해서 4
x=x.reshape(20,-1)
blue_x=x.mean(axis=0) # 표본평균 구하기

x.shape
blue_x.shape

for i in range(10000):
    a=x[i]-blue_x # 기대값을 모를때는 표본평균으로 대체해서 공식사용 표본분산공식은 1/(n-1) *(시그마 x_표본 - 기대값)**2
    a
a.shape
aa=a**2
s_2=[]
s_1=[]
for i in range(10000):
    s2=(1/(len(x)-1)) *aa
    s1=s_1=(1/(len(x))) *aa
    s_2.append(s1)
    s_1.append(s2)
s_2    




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
s_2=[]
s_1=[]
for k in range(10000):
    x=uniform.rvs(loc=3,scale=4,size=20)
    x_mean=np.mean(x)
    a= (x-x_mean)**2
    aa= sum(a)
    s2=(1/(len(x)-1)) *aa
    s1=(1/(len(x))) *aa
    s_2.append(s2)   
    s_1.append(s1) 
s_2    
#x[0]-blue_x 

plt.hist(s_2, color = "blue")

var=np.var(x)
#기대값 표현
plt.axvline(x=var,color="green",linestyle='-',linewidth=2)
plt.show()
plt.clf()


plt.hist(s_1, color = "red")
plt.axvline(x=var,color="green",linestyle='-',linewidth=2)
plt.show()
plt.clf()


np.mean(s_2)
np.mean(s_1)
var

#표본평균의 평균을 내었을때 s_2가 실제 분산과 더 가까우므로 s_2를 쓰는것이 맞다





