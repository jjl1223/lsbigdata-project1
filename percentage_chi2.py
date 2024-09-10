import numpy as np
import pandas as pd

mat_a=np.array([14,4,0,10]).reshape(2,2)

mat_a

#귀무가설: 두 변수 독립
#대립가설 : 두변수가 독립 x

from scipy.stats import chi2_contingency

chi2,p,df,expected=chi2_contingency(mat_a)

chi2.round(3) # 검정 통계량
p.round(4) # p-value

np.sum((mat_a-expected)**2/expected)
# expected 기대빈도

#유의수준 0.05이라면,
#p값이 0.05보다 작으므로 귀무가설을 기가
# 즉 , 두 변수는 독립이 아니다
# X~ chi2(1) 일때, p(X>12.6)=?
from scipy.stats import chi2

1-chi2.cdf(12.6,df=1) # 멀리떨어질수록 귀무가설 기각


#귀무가설: 정당 지지와 헨드폰 사용유무는 독립이다
#대립가설 : 정당지지와 헨드폰 사용유무는 독립이 아니다
# X~ chi2(1) 일때, p(X>15.556)=?
1-chi2.cdf(12.6,df=1)

a=pd.DataFrame({
    "정당지지":["진보","중도","보수"],
    "헨드폰":[49,15,32],
    "유선전화":[47,27,30]

})

chi2,p,df,expected=chi2_contingency(a.iloc[:,1:])
chi2 #검정통계량 
p #pvalue
#p_value가 0.05보다 작으므로 귀무가설을 기각한다

expected # 이게 기대빈도
# 기대빈도가 5이상이여야 신뢰할수있다
from scipy.stats import chisquare
import numpy as np
observed = np.array([13,23,24,20,27,18,15])
expected = np.repeat(20,7)
#그동안은 데이터만 넣고 기대빈도는 넣지 않았는데 여기서는 기대빈도도 넣었다
statistic,p_value=chisquare(observed,f_exp=expected)


print("Test statistice: ",statistic.round(3))

print("P_value: ",p_value.round(3))

#지역별 후보 지지율
#귀무가설: 선거구별 지지율이 동일하다
mat_b=np.array([[176,124],
                [193,107],
                [159,141]])

mat_b

hi2,p,df,expected=chi2_contingency(mat_b)
chi2 #검정통계량 
p #pvalue
#p_value가 0.05보다 작으므로 귀무가설을 기각한다

expected # 기대빈도 5이상 신뢰가능