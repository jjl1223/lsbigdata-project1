import numpy as np
import pandas as pd

tab3=pd.read_csv("data/tab3.csv")
tab3


tab1=tab3[["id","score"]]
tab1.reset_index(drop=False,inplace=True)
tab1
tab1=tab1.drop(["id"],axis=1)
tab1.rename(columns={"index" : 'id'},inplace=True)

tab2=tab1.assign(gender=["female"]*7+["male"]*5)
tab2

# 표본 t 검정 (그룹 1개)
#귀무 가설 vs . 대립가설
# H0: mu = 10 vs. Ha:mu!= 10
#유의 수준 5%로 설정


from scipy.stats import ttest_1samp
# t value pvalue 구해주는 함수
result = ttest_1samp(tab1["score"], popmean=10, alternative='two-sided')
t_value=result[0] # t 검정 통계량
p_value=result[1] # 유의 확률(p-value)
tab1["score"].mean() # 표본평균

result.pvalue # 유의 확률(p-value)
result.statistic # t 검정 통계량
result.df # 자유도
# 신뢰구간 양측 값 구해주는 함수
#95% 신뢰구간 구하기
ci=result.confidence_interval(confidence_level=0.95) 
ci[0]
ci[1]



# 2표본 t검정 (그룹2)

## 귀무가설 vs 대립가설
## H0: mu_m:mu_f vs. Ha: mu_m>=mu_f
## 유의수준 1%로 
# 분산 같을 경우: 독립 2표본 t검정
# 분산 다를경우 : 웰치스 t 검정

from scipy.stats import ttest_ind
f_tab2=tab2[tab2["gender"]=="female"]
m_tab2=tab2[tab2["gender"]=="male"]


# alternative="less"의 의미는 대립설이 첫번째 입력그룹의 평균이 두번째 입력 그룹 평균보다 작다고 설정된 경우를 나타냄
# equal_var=True 두 그룹의 분산이 같다
result=ttest_ind(m_tab2["score"],f_tab2["score"],alternative="greater",equal_var=True)
ttest_ind(f_tab2["score"],m_tab2["score"],alternative="less",equal_var=True)

result.pvalue # 유의 확률(p-value)
result.statistic # t 검정 통계량


# 3 
# 피벗테이블로 같은 아이디 합치기
tab3
tab3_data=tab3.pivot_table(index="id",columns="group",values="score") # id group이 이상하게 묶이는 문제발생
tab3_data["score_diff"]=tab3_data["after"]-tab3_data["before"]
tab3_data


# 대응표본 t 검정 (짝지을 수 있는 표본)
## 귀무가설 vs. 대립가설
## H0: mu_before = mu_after vs. Ha: mu_after > mu_before
## H0: mu_d = 0 vs. Ha: mu_d > 0
## mu_d = mu_after - mu_before
## 유의수준 1%로 설정


#mu_d에 대응하는 표본으로 변환
#pivot table id를 기준으로 일치되는값들을 묶고  group에 일치되는 값들을 묶는다
tab3_data = tab3.pivot_table(index="id",columns="group",values="score").reset_index() # 그래서 reset index추가
tab3_data["score_diff"]=tab3_data["after"]-tab3_data["before"]
tab3_data

from scipy.stats import ttest_1samp
result = ttest_1samp(tab3_data["score_diff"], popmean=0, alternative='greater')
t_value=result[0] # t 검정통계량
p_value=result[1] # 유의확률 (p-value)
t_value; p_value

#위에 pivot table 반대로 풀어서 원상복귀하기
long_form = tab3_data.reset_index().melt(id_vars='id', value_vars=['before', 'after'], var_name='group', value_name='score')
#연습1
import seaborn as sns
df = pd.DataFrame({"id" : [1,2,3],
                    "A"  :[10,20,30],
                    "B" : [40,50,60] 
                        })
                        
df_long=df.melt(id_vars="id",
        value_vars=["A","B"],
        var_name="group",
        value_name="score"
        
        )
        
df_long.pivot_table(columns="group",values="score")   # index가 없으면 group이 같으면 같다고 이해해서 평균값으로 내준다
df_long.pivot_table(columns="group",values="score",aggfunc="max") # 평균값대신 다른것을 하고 싶으면 aggfunc에 다른것을 해주면 된다

#연습2                        
import seaborn as sns
tips=sns.load_dataset("tips")


tips.reset_index(drop=False).pivot_table(index="index",columns="day",values="tip").reset_index()
#위와 아래 결과값이 같고 대신 아래는 index가 안남아있다
tips.pivot(columns="day",values="tip")


tips.reset_index(drop=False).pivot_table(index=["index"],columns="day",values="tip").reset_index()






