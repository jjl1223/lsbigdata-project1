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
#F_x =P(X=x)
bernoulli.rvs(p=0.3,size=1)

bernoulli.rvs(p=0.3) + bernoulli.rvs(p=0.3)

binom.rvs(n=0.2,p=0.3,size=1)


binom.pmf(k=1,n=10,p=0.26) # 10개중에 1이 하나 나올 확률

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



#cdf : cumulative dist.function
#(누적확률분포 함수)
#F_x =P(X<=x)
#binom.cdf(4, n=30, p=0.26)

binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26)

binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)

x=np.arange(31)
prob_x=binom.pmf(x,n=30,p=0.26)

plt.clf()
sns.barplot(prob_x,color="blue")
sns.barplot(x=x,y=prob_x,color="red")
plt.show()


x_1= binom.rvs(n=30,p=0.26,size=1)
#add a point at(2,0)  포인트 위치 지정 색 zorder:어디 영역에 할지 s: 마커사이즈
plt.scatter(2, 0 ,color="red",zorder=100,s=300)
plt.show()



plt.clf()
x_1= binom.rvs(n=30,p=0.26,size=5)
sns.barplot(prob_x,color="blue")
plt.scatter(x_1, np.array([0.03,0.04,0.05,0.06,0.07]) ,color="red",zorder=100,s=100) #x가 늘어난 만큼 y값도 늘려주면 다 찍어줌
plt.scatter(x_1, np.repeat(0.03,5),color="red",zorder=100,s=100) #x가 늘어난 만큼 y값도 늘려주면 다 찍어줌

plt.axvline(x=7.8,color="green",linestyle='--',linewidth=2) # 마커 찍기
plt.show()


#x~ B(n,p)
#앞면이 나올 확률이 p인ㄷ 동전을 n번 던져서 나온 앞면의 수

#pmf,cdf,rvs,ppf

binom.ppf(0.5,n=30,p=0.26) #왼쪽끝부터 확률을 더해서 0.5 가 되는 값 찾아줌 
binom.cdf(7,n=30,p=0.26) #지정한값이하로 나올 확률








# 정규분포함수
def jungu(x,mu,seta):
    sss=1/(seta*(2*math.pi)**0.5)*math.exp((-1)/2*x**2)
    return sss

jungu(0,0,1)
jungu(5,3,4) 
from scipy.stats import norm



norm.pdf(0,loc=0,scale=1) # x,loc=mu,scale=sigma
norm.pdf(5,loc=3,scale=4) #지정된수에 대응하는 y축값(확률밀도함수의 값)  

k=np.linspace(-3,3,100)
y=norm.pdf(k,loc=0,scale=1)
plt.clf()
plt.scatter(k,y,color="red")
plt.show()

y=norm.pdf(k,loc=-3, scale=1) # 뮤(log)는 확률분포의 중심을 뜻한다(평균)
plt.plot(k,y,color="black")
plt.show()
plt.clf()

 norm.cdf(0,loc=0,scale=1) # 왼쪽끝에서 지정된수까지 도달했을때 면적(적분한것)
 norm.cdf(-2,loc=0,scale=1)

 norm.cdf(-0.54,loc=0,scale=1)

 a=norm.cdf(3,loc=0,scale=1) -norm.cdf(1,loc=0,scale=1)
 
 1-a
 
#정규분포 x~ N(3,5^2)
#P(3<x<5) =? 15.54
 
  norm.cdf(5,loc=3,scale=5) -   norm.cdf(3,loc=3,scale=5)
# 위 확률변수에서 표본 1000개 뽑아보기 
 
vvv=norm.rvs(loc=3,scale=5,size=1000) 
 
a=((3<vvv) & (vvv<5))
sum(a)/1000
#평균 0 ,표준편차:1
# 표본 1000개 뽑아서 0보다 작은 비율 확인

vv=norm.rvs(loc=0,scale=1,size=1000) 

 a=0>vv
sum(a)/1000
np.mean(vv<0)

x=norm.rvs(loc=3,scale=2,size=1000) 
x
sns.histplot(x,stat='density') # y축은 빈도수 stat='density'y축을 빈도대신 확률로 바꿔준다
x_min,x_max=(x.min() , x.max())
x_values = np.linspace(x_min,x_max,100)
pdf_values = norm.pdf(x_values,loc=3,scale=2)
plt.scatter(x_values,pdf_values,color="red",linewidth=2) # scatter는 점으로 plot은 선으로
plt.show()
plt.clf()

#숙제1
#정규분포 pdf 값을 계산하는 자신만의 
#파이썬 함수를 정의하고 정규분포 mu=3 sigma=2의 pdf를 그릴것
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def jungu(x,mu,sigma):
    sss=(1/(sigma*((2*math.pi)**0.5)))* math.exp( ( -0.5) *( ((x-mu)/sigma) **2) )
    return sss

jungu(0,3,2)
pdf_values=[]

x_values = np.linspace(-5,10,100)
for x in np.linspace(-5,10,100):
    pdf_values.append(jungu(x,3,2))

plt.plot(x_values,pdf_values,color="red",linewidth=2) # scatter는 점으로 line은 선으로
plt.show()
plt.clf()



#2. 파이썬 scipy 패키지 사용해서 다음과 같은 확률을 구하시오
#x ~ N(2,3^2)
#1) P(x<3)
#2 P(2<x<5)
#3 P(x<3 or x>7)
from scipy.stats import norm

norm.cdf(3,2,3**2)

norm.cdf(5,2,3**2)-norm.cdf(2,2,3**2)

1-(norm.cdf(7,2,3**2)-norm.cdf(3,2,3**2)) 

#3.LS 빅데이터 스쿨 학생들의 중간고사점수는 평균이 30이고, 분산이 4인 정규분포를 따른다
#상위 5%에 해당하는 학생의 점수는?

0.05
norm.ppf(0.95,30,4) # 상위 5% 즉 하위 95%값을 반환해줌 #정답

norm.cdf(36.5794,30,4) # 왼쪽에서부터 지정한값까지 나올 확률을 다 더해줌(면적)



 
