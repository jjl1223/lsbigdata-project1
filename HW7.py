from scipy.stats import norm
import pandas as pd
import seaborn as sns    
import matplotlib.pyplot as plt
import numpy as np    
1-norm.cdf(18,loc=10,scale=1.96)
# apd 57P문제
#귀무가설: 현대자동차 에너지 소비효율은 16이상이다
s=[15.078, 15.752, 15.549, 15.56, 16.098, 13.277, 15.462, 16.116, 15.214, 16.93, 14.118, 14.927,
15.382, 16.709, 16.804]

s_mean=np.mean(s)
s_std=np.std(s,ddof=1) # 표준편차 구핼때는 n-1이므로 ddof=1로 해주기
s_n=len(s)

t_s=(16-s_mean)/(s_std/np.sqrt(s_n))


from scipy.stats import t

p_value=1-t.cdf(t_s,df=s_n-1) #16이상이므로 단측 cdf t.cdf 왼쪽에서부터 지정한 값까지의 면적
print("P_value는 ",p_value)
#0.043로 0.001보다 크므로 귀무가설을 체택한다

# 95%신뢰구간
#t.ppf 하위 %값을 반환해줌 ex)t.ppf(0.975,df=s_n-1) 하위 97.5%까지의 값 즉 상위 2.5% 값을 반환해준다
16 + t.ppf(0.975,df=s_n-1) *s_std / np.sqrt(s_n)
16 - t.ppf(0.975,df=s_n-1) *np.std(s,ddof=1) / np.sqrt(s_n)


