import pandas as pd
import numpy as np
import matplotlib_inline as plt
from scipy.stats import binom
from scipy.stats import norm
a=pd.DataFrame({
    "x":[2,3,4,5,6,7,8,9,10,11,12],
    "P":[1/36,2/36,3/36,4/36,5/36,6/36,5/36,4/36,3/36,2/36,1/36]

})

E_x=sum(a["x"]*a["P"])

binom.pmf(0,3,0.7)

# x_bar ~ N(30,4^2/8)
a= norm.cdf(29.7,loc=30,scale=np.sqrt(4**2/8))
b= norm.cdf(28,loc=30,scale=np.sqrt(4**2/8))

a-b
#표준화 사용방법
mean=30
s_var=4/np.sqrt(8) #표준편차
right_x=(29.7-mean)/s_var #표준화
left_x= (28-mean)/s_var

a= norm.cdf
# 카이제곱 분포
from scipy.stats import chi2
k= np.linspace(0,20,100)
y = chi2.pdf(k,df=7) # k가 들어가는 수 df는 자유도 v
plt.plot(k,y,color="black")
