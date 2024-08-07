# norm 정규분포함수 다양한 값 기능 신뢰구간 분산 구하기 t분포도 마찬가지로 하고 각조건에 따른 맞는 신뢰구간 구하기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    
import seaborn as sns    

from scipy.stats import norm

norm.cdf(0.25,loc=3,scale=7) # 왼쪽에서부터 지정한값까지 나올 확률을 다 더해줌(면적) loc는 평균=기대값 scale은 분산
norm.ppf(0.95,30,4) # 상위 5% 즉 하위 95%값을 반환해줌 #정답
norm.pdf(0.25,loc=3,scale=7) #지정된수에 대응하는 y축값(확률밀도함수의 값) 



norm.pdf(0.25,loc=0,scale=1) #지정된수에 대응하는 y축값(확률밀도함수의 값) 
#N(3,7**2) 쓸떄는 (평균,분산)
norm.ppf(0.25,3,7) 
z=norm.ppf(0.25,0,1)


x=3+z*7 # norm.ppf(0.25,3,7) 이거랑 같음 #3 평균 7이 표준편차


norm.cdf(5,loc=3,scale=7) 
norm.cdf(2/7,loc=0,scale=1) 


norm.ppf(0.974,loc-0,scale=1)



z=norm.rvs(loc=0, scale=1, size=1000)
z

x=z*np.sqrt(2) +3
sns.histplot(x, stat="density")
sns.histplot(z, stat="density")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
pdf_values2 = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.plot(z_values, pdf_values2, color='red', linewidth=2)



plt.show()
plt.clf()


# 표본표준편차 나눠도 표준정규분포가 될까?
#1.
x=norm.rvs(loc=5, scale=3, size=20)
s=np.std(x, ddof=1)
s
# s_2=np.var(x, ddof=1)

#2.
x=norm.rvs(loc=5, scale=3, size=1000)

# 표준화
z=(x - 5)/s
# z=(x - 5)/3
sns.histplot(z, stat="density", color="grey")

# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()
#결론: 되지 않는다

#표준화: z= (x-mu) / 표준편차  :신뢰구간 값으로 x값을 바꿔주는것



# t-분포에 대해서 알아보자
# x~t(df)
#연속형 확률변수이고 정규분포랑 비슷하게 생김
# 종모양 대칭분포 중심 0
# 모수 df: 자유도라고 부름 분산에 영향을 미침(퍼짐을 나타내는 모수)
# df이 작으면 분산 커짐

#t.pdf 지정된 수에 대응하는 y값
#t.ppf 하위 %값을 반환해줌 ex) 
#t.cdf 왼쪽에서부터 지정한 값이 다 나온다
#t.rvs 랜덤값 만들기
from scipy.stats import t
t_values= np.linspace(-4,4,100)
pdf_valuse = t.pdf(t_values,df=30)
plt.plot(t_values,pdf_valuse,color="red",linewidth=2)

# 꼬리가 길다(fat tail) = 양쪽 끝에 수가 나올확률이 더 높은 경우


#표준 정규분포 겹치기
pdf_valuse=norm.pdf(t_values, loc=0,scale=1)
plt.plot(t_values,pdf_valuse,color="black",linewidth=2)


plt.show()
plt.clf()

#자유도 즉 df가 커지면 커질수록 정규분포와 매우 비슷해진다.


#x ~ ?(mu,sigma^2) 
#X bar ~ N(mu,sigma^2/n)
#X bar ~= N(x_bar,s^2/n) 자유도 n-1인 t분포

x=norm.rvs(loc=15,scale=3,size=16,random_state=42)
x
n=len(x)
#df=degree of freedom
#모분산을 모를때: 모 평균에 대한 95%신뢰구간을 구해보자
x_bar + t.ppf(0.975,df=n-1) *np.std(x,ddof=1) / np.sqrt(n)
x_bar - t.ppf(0.975,df=n-1) *np.std(x,ddof=1) / np.sqrt(n)
# HW6은 정상적으로 구한 신뢰구간이 아니다 

#모분산을 알때 : 모 평균에 대한 95%신뢰구간을 구해보자 여기서는 모분산이(3^2)
x_bar + norm.ppf(0.975,loc=0,scale=1) *3 / np.sqrt(n)
x_bar - norm.ppf(0.975,loc=0,scale=1) *3 / np.sqrt(n)

#adp 교재 43쪽 스튜던트 정리




