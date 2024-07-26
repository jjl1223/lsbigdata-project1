import numpy as np    
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform

uniform.rvs(2,6,size=1)
# loc 시작점 scale 길이
#uniform.pdf(x, loc=0, scale=1)
#uniform.cdf(x, loc=0, scale=1)
#uniform.ppf(x, loc=0, scale=1)
#uniform.rvs(loc=0,scale=1,size=None,random_state)

k= np.linspace(0,8,100)
y=uniform.pdf(k,loc=2,scale=4)
plt.plot(k,y,color= "black")

plt.show()
plt.clf()

uniform.cdf(3.25,loc=2,scale=4)


uniform.cdf(8.39,loc=2,scale=4)-uniform.cdf(5,loc=2,scale=4)

uniform.ppf(0.93,loc=2,scale=4)


#신뢰구rks
#x bar ~N(mu,sigma^2/n)
#x bar ~N(4,1.33333/20)

from scipy.stats import norm
x_values = np.linspace(3,5,100)
pdf_values = norm.pdf(x_values,loc=4,scale=np.sqrt(1.333333/20))
plt.plot(x_values,pdf_values,color="red",linewidth=2) # scatter는 점으로 plot은 선으로

#표본평균(파란벽돌 점찍기)
blue_x=uniform.rvs(loc=2,scale=4,size=20).mean()


a=blue_x+0.665 # a=blue_x+1.96*np.sqrt(1.333333/20)
b=blue_x-0.665 # a=blue_x-1.96*np.sqrt(1.333333/20)
#점찍기
plt.scatter(blue_x,0.002,color="blue",zorder=10,s=10) #s 마커사이즈 zoredr 한글의 맨앞으로 맨뒤로 기능 숫자가크면 앞으로
#영역표시(라인)
plt.axvline(x=a,color="blue",linestyle="--",linewidth=1)
plt.axvline(x=b,color="blue",linestyle="--",linewidth=1)


#기대값 표현
plt.axvline(x=4,color="green",linestyle='-',linewidth=2)
plt.show()
plt.clf()

norm.ppf(0.025,loc=4,scale=np.sqrt(1.333333/20))
norm.ppf(0.975,loc=4,scale=np.sqrt(1.333333/20))

4-norm.ppf(0.005,loc=4,scale=np.sqrt(1.333333/20))
4-norm.ppf(0.995,loc=4,scale=np.sqrt(1.333333/20))




#표본 20개 뽑기

x=uniform.rvs(loc=2,scale=4,size=1000)
x.mean()
x=uniform.rvs(loc=2,scale=4,size=1000,random_state=42) # random_state는 모든 경우에서 랜덤값이 일정하게 나오게 해준다
x.mean()

x=uniform.rvs(loc=2,scale=4,size=20*1000,random_state=42) # random_state는 모든 경우에서 랜덤값이 일정하게 나오게 해준다
x=x.reshape(-1,20) # -1은 값 대신 맞춰주는것
x.shape
x.mean()
blue_x=x.mean(axis=1)
blue_x

import seaborn as sns
sns.displot(blue_x,stat="density")

plt.show()

uniform.var(loc=2,scale=4) #분산 구하기
uniform.expect(loc=2,scale=4) #기대값 구하기

x_min,x_max=(x.min() , x.max())
x_values = np.linspace(x_min,x_max,100)
pdf_values = uniform.pdf(x_values,loc=3,scale=np.sqrt(1.333333/20))
plt.plot(x_values,pdf_values,color="red",linewidth=2) # scatter는 점으로 plot은 선으로

plt.show()
plt.clf()
