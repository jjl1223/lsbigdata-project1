import numpy as np 
import pandas as pd



temp_frame = pd.DataFrame({"year":2011,"income":630})
#E[x]
np.arange(33).sum()/33

np.arange(33)-16

np.arange(33)**2

np.arange(33)

(np.arange(33)-16)**2

np.unique((np.arange(33)-16)**2)*(2/33)
#E[(x-E(x)^2)]
sum(np.unique((np.arange(33)-16)**2)*(2/33))


# E[x^2]

sum(x**2 * (1/33))

#var(x) = E[x^2] - (E[x])^2
sum(x**2 * (1/33)) - 16**2

#기대값
a=np.arange(4) #x

b=np.arange(4)**2 #x^2

c=np.array([1/6,2/6,2/6,1/6])
Exx=a*b# Ex^2
Ex=a*c# Ex
#분산
#var(x) = E[x^2] - (E[x])^2
sum(b*c)+sum(a*c)**2


#E[(x-E(x)^2)]
sum((a-Ex) **2) *c)



a=np.arange(99)
a_1=np.arange(49,0,-1)
b=np.arange(99)**2

b_1=np.arange(49,0,-1) **2
c=np.arange(51)
d=c[1:51]/2500
e=np.arange(49,0,-1)/2500
(b[1:51]*d) +( b_1*e)  - ( a[1:51]*d) + (a_1*e)



sa=(b[1:51]*d)
sb=( b_1*e)
sc=( a[1:51]*d)
sd=(a_1*e)

sum(sa)+sum(sb)-sum(sc)-sum(sd)

from scipy.stats import bernoulli # scipy라는 모듈에서 stats라는 패키지를 실행한다. 그패키지에서 bernoulli라는 함수실행

#확률질량함수(pmf)
#확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
#bernoulli.pmf(k,p) 교재 27p
#p(x=1)이 나올 확률 0.3

bernoulli.pmf(1, 0.3)
#p(x=0)이 나올 확률 0.3
bernoulli.pmf(0, 0.3)

# 이항분포 x~p (x=k | n,p)
#n: 베르누이 확률변수
#p: 1이 나올 확률
# binom.pmf(k,n,p)

from scipy.stats import binom

binom.pmf(0,n=2,p=0.3)

binom.pmf(1,n=30,p=0.3)

result=[binom.pmf(x,n=30,p=0.3) for x in range(31)] #깂 list로 받기

binom.pmf(np.arange(31),n=30,p=0.3) # 값을 자동으로 array로 반환

for x in range(54):
    
    
df=np.arange(55)

np.cumprod(df[1:56])[-1] #연산범위 초과 오버플로우

import math
math.factorial(54)/(math.factorial(26)*math.factorial(28))
math.comb(54,26) #확률 nCk (조합)
 
math.log(math.factorial(54))
a=np.log(np.arange(1,55))
b=sum(a)
sum(np.log(np.arange(1,55)))
math.e**b

#log(54!)-(log(26!)+log(28!))

c=sum(np.log(np.arange(1,55))) -( sum(np.log(np.arange(1,27)))+sum(np.log(np.arange(1,29))))

np.exp(c)

#=================================================================================
#3c0 0.3 *(1-0.3)**#
#확률 조합함수
math.comb(2,0) * (0.3 **0) * ((1-0.3)**2)
binom.pmf(0,n=2,p=0.3)

math.comb(2,1) * (0.3 **1) * ((1-0.3)**1)
binom.pmf(1,n=2,p=0.3)
math.comb(2,2) * (0.3 **2) * ((1-0.3)**0)
binom.pmf(2,n=2,p=0.3)

# x~ B(n=10,p=0.36)
#x=4일때
binom.pmf(4,n=10,p=0.36)

# x~ B(n=10,p=0.36)
#x<=4일때


sum(binom.pmf(range(0,5),n=10,p=0.36))
sum(binom.pmf(np.arange(0,5),n=10,p=0.36))
binom.pmf(np.arange(0,5),n=10,p=0.36).sum()

sum(binom.pmf(np.arange(3,9),n=10,p=0.36))

np.sum(binom.pmf(np.arange(3,9),n=10,p=0.36)
# x~ B(30,0.2)
#확률 변수 x가 4보다 작거나, 25보다 크거나 같은 확률을 구하시오



binom.pmf(np.arange(0,4),30,0.2).sum() +  binom.pmf(np.arange(25,31),30,0.2).sum()


1-binom.pmf(np.arange(4,25),30,0.2).sum()

#rvs 함수 (random variates sample)
# 표본 추출 함수
# x~ Bernulli(p=0.3) p= 1이 나올 확률
bernoulli.rvs(p=0.3,size=1)

bernoulli.rvs(p=0.3) + bernoulli.rvs(p=0.3)

binom.rvs(n=0.2,p=0.3,size=1)


binom.pmf(k=1,n=10,p=0.26)

# 베르누이 확률변수 기대값은 p p가 1이나올 확률이니까 1*p+0*(1-p)이므로
# 이항분포 확룰변수 x의 기대값 = np
import seaborn as sns

b=binom.pmf(k=(np.arange(0,31)),n=30,p=0.26)
k=(np.arange(0,31))

sns.barplot(x=k,y=b)
import matplotlib.pyplot as plt
plt.show()

a=pd.DataFrame({ 'k' : np.arange(0,31),
               'pre' : b
               })
plt.clf()
sns.barplot(data=a,x="k",y='pre')

plt.show()
