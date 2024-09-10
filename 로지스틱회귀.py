import pandas as pd
import numpy as np
# 교재 10장

# 워킹 디렉토리 설정
# import os
# cwd=os.getcwd()
# parent_dir = os.path.dirname(cwd)
# os.chdir(parent_dir)

admission_data= pd.read_csv("./data/admission.csv")

print(admission_data.shape)

#GPA: 학점
#GRE : 대학원 입학시험(영어,수학)

# 합격을 한 사건 : Admit
# Admit의 확률 오즈(Odds)는?
# P(admit)=합격인원 / 전체학생

p_hat= admission_data['admit'].mean()
p_hat / (1-p_hat) # 확률 오즈비 구하기

# p(A) : 0.5 -> 오즈: 무한대에 가까워짐
# p(A) : 0.5 -> 오즈: 1
# p(A) : 0.5보다 작은 경우 -> 오즈:0에 가까워짐
# 확률의 오즈 : 갖는 값의 범위: 0~무한대

unique_ranks = admission_data['rank'].unique()

grouped_data = admission_data \
    .groupby('rank', as_index=False) \
    .agg(p_admit=('admit', 'mean'), )
grouped_data['odds'] = grouped_data['p_admit'] / (1 - grouped_data['p_admit'])
print(grouped_data)

#오즈비가 3일때
# p(A)?
# p(A)/1-p(A) =3

#admission 데이터 산점도 그리기
# x: gre, y: admit
# admission_data
import seaborn as sns

sns.scatterplot(data=admission_data,x="gre",y="admit")
sns.scatterplot(data=admission_data,x="rank",y="admit")
#겹쳐지는 점도 표시하기 위해서 사용
sns.stripplot(data=admission_data,
              x='rank', y='admit', jitter=0.3, alpha=0.3)

sns.scatterplot(data=grouped_data,
              x='rank', y='p_admit')

sns.regplot(data=grouped_data, x='rank', y='p_admit')         



odds_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean')).reset_index()
odds_data['odds'] = odds_data['p_admit'] / (1 - odds_data['p_admit'])
odds_data['log_odds'] = np.log(odds_data['odds'])
print(odds_data)

sns.regplot(data=odds_data, x='rank', y='log_odds')


import statsmodels.api as sm
# log_odds~가 분석하고 싶은대상(독립변수) rank가 종속변수 
# 선형회귀분석 ols: 차이 제곱이 최소인걸로 하기
model = sm.formula.ols("log_odds ~ rank", data=odds_data).fit()
print(model.summary())

# 더미코드 처리(텍스트인 경우는 자동으로 더미처리)
admission_data['rank'] = admission_data['rank'].astype('category') 
admission_data['gender'] = admission_data['gender'].astype('category')
# admit의 로그오즈에 대한 모델 만들기 admit(독립변수) ~나머지 전부 종속변수
model = sm.formula.logit("admit ~ gre + gpa + rank + gender", data=admission_data).fit()

print(model.summary())


입학할 확률의 오즈가 
np.exp(0.7753)

# 여학생
# GPA: 3.5
# GRE: 500
# Rank: 2

# 합격 확률 예측해보세요!
# odds = exp(-3.408 + -0.058 * x1 + 0.002 * x2 + 0.775 * x3 -0.561 * x4)
my_odds=np.exp(-3.408 + -0.058 * 0 + 0.002 * 500 + 0.775 * 3.5 + -0.561 * 2)
my_odds / (my_odds+1) # 합격 확률: 0.306

# 여학생
# GPA: 3
# GRE: 450
# Rank: 2
# 합격 확률? odds?

my_odds=np.exp(-3.408 + -0.058 * 0 + 0.002 * 450 + 0.775 * 3 + -0.561 * 2)
my_odds / (my_odds+1) # 합격 확률: 0.2133


print(model.summary())
# 여기서 나오는 coef는 각 변수의 계수 intercept는
# y절편 p>ㅣzㅣ는 p-value= 유의확률
# 귀무가설이 통계적으로 유의하지 않다이다
# 정확히 그 변수의 계수(베타)가 0이다이다
# 그래서 귀무가설을 기각하지 못하면 그 종속변수는
# 독립변수에 유의하지 못하다는 뜻이다
# 근데 이때 통계적으로 유의하지 않다고 해서 
# 그 변수를 뺴고 회귀식을 쓰는것이 아니다
# 단지 그변수를 해석하지 못하는 것 뿐이다
# 그래서 기각하지 못하는 변수를 뺴고
# 다시 회귀모델을 돌려서 사용해도 된다
# 근데 이거는 그 변수를 쓰는것보다 
# 뺴고 다시돌린 회귀모델이 좋을거라고 가정하고 하는것이다


from scipy.stats import norm
1-norm.ppf(0.025, loc=0, scale=1)
2*(1-norm.cdf(2.123, loc=0, scale=1))
2*norm.cdf(-2.123, loc=0, scale=1)
#검정 통계량 구하는법(LL-Null-Log-Likelihood)
stat_value=-2*(-249.99 - (-229.69))

from scipy.stats import chi2
# 검정 통계량 이용해서 p-value 구하기
1-chi2.cdf(stat_value, df=4) # df=변수갯수