import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
raw_welfare = pd.read_spss('data/koweps/Koweps_hpwc14_2019_beta2.sav')

welfare = raw_welfare.copy()

welfare
# 칼럼 알아보기 쉬운 이름으로 바꿔주기
welfare= welfare.rename(
    
    columns = { 
        'h14_g3' : 'sex',
        'h14_g4' : "birth",
        'h14_g10' : "marriage_type",
        'h14_g11' : "religion",
        'p1402_8aq1' : "income",
        'h14_eco9' : "code_job",
        "h14_reg7" : "code_region"
        
        }
)

#쓰는 데이터만 가져오기
welfare=welfare[["sex", "birth", "marriage_type",
                "religion", "income", "code_job", "code_region"]]
welfare.shape


welfare["sex"] = np.where(welfare["sex"]==1, 'male', 'female' )


from scipy.stats import norm
z_005=norm.ppf(0.95,loc=0,scale=1) #z값 구하기
# 성별데이터 따로 만들기
female_data=welfare.query('sex=="female"')
male_data=welfare.query('sex=="male"')

fe=female_data.dropna(subset='income')["income"] #female income 데이터
me=male_data.dropna(subset='income')["income"] #male income 데이터

fe_v=np.var(fe,ddof=1) #n-1로 나눈 값(표본분산)
me_v=np.var(me,ddof=1)

fe_n=fe.count() #표본수 세기
me_n=me.count()
#평균값 만들기 # 줄바꿈은 한칸뛰어있어야 하며 뒤에 스페이스가 들어가 이쓰면 안된다
fe_mean_d=female_data.dropna(subset="income")\   
                   .groupby("sex",as_index=False)\
                  .agg(mean_income = ("income","mean"))
me_mean_d=male_data.dropna(subset="income")\   
                  .groupby("sex",as_index=False)\
                  .agg(mean_income = ("income","mean"))
fe_mean=fe_mean_d['mean_income'][0]                  
me_mean=me_mean_d['mean_income'][0]                     
        
#신뢰구간 연산                    
female_a=fe_mean +z_005 *np.sqrt(fe_v) /np.sqrt(fe_n) 
female_b=fe_mean  -z_005 *np.sqrt(fe_v) /np.sqrt(fe_n) 

male_a=me_mean +z_005 *np.sqrt(me_v) /np.sqrt(me_n) 
male_b=me_mean  -z_005 *np.sqrt(me_v) /np.sqrt(me_n) 

#전체 그래프 그리기
welfare.dropna(subset="income").shape
sex_income=welfare.dropna(subset="income")\
       .groupby("sex",as_index=False)\
       .agg(mean_income = ("income","mean"))
sns.barplot(data=sex_income,x="sex",y="mean_income", hue="sex")
plt.axhline(y = female_a, color = 'black', linestyle = '-', linewidth = 2)
plt.axhline(y = female_b, color = 'black', linestyle = '-', linewidth = 2)
plt.axhline(y = male_a, color = 'brown', linestyle = '-', linewidth = 2)
plt.axhline(y = male_b, color = 'brown', linestyle = '-', linewidth = 2)
plt.show()
plt.clf()

#그리기용 데이터 프레임만들기
fe_draw=pd.DataFrame({'sex' :["female"],
                   'mean_income' : [fe_mean]   })
me_draw=pd.DataFrame({'sex' :["male"],
                   'mean_income' : [me_mean]   })
#그래프 따로 그리기
#여자 그래프
sns.barplot(data=fe_draw,x="sex",y="mean_income", hue="sex")
plt.axhline(y = female_a, color = 'black', linestyle = '-', linewidth = 2)
plt.axhline(y = female_b, color = 'black', linestyle = '-', linewidth = 2)
plt.show()
plt.clf()
#남자 그래프
sns.barplot(data=me_draw,x="sex",y="mean_income", hue="sex")
plt.axhline(y = male_a, color = 'black', linestyle = '-', linewidth = 2)
plt.axhline(y = male_b, color = 'black', linestyle = '-', linewidth = 2)
plt.show()
plt.clf()






# 숙제: 위 그래프에서 각 성별 95% 신뢰구간 계산후 그리기
#위 아래 검정색 막대기로 표시

#norm.ppf() 사용해서 그릴것. 모분산은 표본 분산을 사용해서 추정
# 잔재들
welfare.dropna(subset="income").shape
sex_income=welfare.dropna(subset="income")\
       .groupby("sex",as_index=False)\
       .agg(mean_income = ("income","mean"))
       
sex_income


import seaborn as sns

sns.barplot(data=sex_income,x="sex",y="mean_income", hue="sex")


plt.show()



bb=welfare["sex"].value_counts() # 각성별 전체 수 세기


female_n=bb.iloc[0] #female male 표본수 카운트
male_n=bb.iloc[1]
female_mean=sex_income['mean_income'][0] #평균값 빼오기 0행이 여자 1행이 남자
male_mean=sex_income['mean_income'][1]


