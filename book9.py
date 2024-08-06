import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
raw_welfare = pd.read_spss('data/koweps/Koweps_hpwc14_2019_beta2.sav')




welfare = raw_welfare.copy()

welfare
welfare.shape
welfare.info()
welfare.describe()

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


welfare=welfare[["sex", "birth", "marriage_type",
                "religion", "income", "code_job", "code_region"]]
welfare.shape

welfare["sex"].dtypes
welfare["sex"].value_counts()
welfare["sex"].isna().sum()
welfare["sex"] = np.where(welfare["sex"]==1, 'male', 'female' )

welfare["sex"] 

welfare["income"].describe()
welfare["income"].isna().sum()

(welfare["income"]>9998).sum()

welfare.dropna(subset="income").shape
sex_income=welfare.dropna(subset="income")\
       .groupby("sex",as_index=False)\
       .agg(mean_income = ("income","mean"))
       
sex_income



import seaborn as sns

sns.barplot(data=sex_income,x="sex",y="mean_income", hue="sex")


plt.show()
plt.clf()




# 숙제: 위 그래프에서 각 성별 95% 신뢰구간 계산후 그리기
#위 아래 검정색 막대기로 표시

#norm.ppf() 사용해서 그릴것. 모분산은 표본 분산을 사용해서 추정


welfare["birth"].describe()
sns.histplot(data=welfare,x="birth")
plt.show()
plt.clf()

welfare["birth"].isna().sum()


welfare=welfare.assign(age = 2024- welfare["birth"] + 1 )
welfare["age"]

sns.histplot(data=welfare,x="age")
plt.show()
plt.clf


# dropna는 결측치 제거하기

age_income = welfare.dropna(subset = "income") \ 
                    .groupby("age",as_index=False) \
                    .agg(mean_income = ('income','mean'))
                    
age_income.head()                    

sns.lineplot(data=age_income , x="age",y="mean_income")
plt.show()
plt.clf()


=welfare.assign(income_na=welfare["income"].isna()) \
             .groupby("age",as_index=False) \
             .agg(n = ("income_na","sum"))

sns.barplot(data=my_df,x="age",y="n")
plt.show()
plt.clf()

sss= welfare.groupby("age",as_index=False)

welfare['age'].head()

welfare= welfare.assign(ageg= np.where(welfare['age']<30 , 'young',
                             np.where(welfare['age']<= 59, 'middle',
                                                            'old')))
                                                            
welfare['ageg'].value_counts()

sns.countplot(data = welfare, x= 'ageg')
plt.show()
plt.clf()      



ageg_income=welfare.dropna(subset='income') \
                   .groupby('ageg',as_index=False) \
                   .agg(mean_income=('income','mean'))
                   
                   
sns.barplot(data = ageg_income, x= 'ageg',y='mean_income')
plt.show()
plt.clf()  
sns.barplot(data= ageg_income,x='ageg',y='mean_income', order= ['young','middle','old'])
  plt.show()
plt.clf()                                                           
                                                            
 
# 나이대별 수입 분석
# 변수쪼개기
a = np.array([1,2,3,4,5,6,7,8]) ## 데이터
cut = pd.cut(a, bins=[0,3,5]) ## 데이터를 최소 최대 구간으로 3등분한다.
cut
welfare.assign(age_group = pd.cut(welfare['age'],bins=[0,9,19,29,39,49,59,69,79,89,99,109,119]),\
labels=['0대','10대','20대','30대','40대','50대','60대','70대','80대','90대','100대']) #안됨


age_group = pd.cut(welfare['age'],bins=[0,10,20,30,40,50,60,70,80,90,100,110]),labels=['0대','10대','20대','30대','40대','50대','60대','70대','80대','90대','100대']



bin_cut=np.array([0,9,19,29,39,49,59,69,79,89,99,109,119])
labels1 =(np.arange(12) *10).astype(str)+"대"


welfare=welfare.assign(age_group = pd.cut(welfare['age'],bins=bin_cut,\
labels=['0대','10대','20대','30대','40대','50대','60대','70대','80대','90대','100대'])

welfare=welfare.assign(age_group = pd.cut(welfare['age'],bins=bin_cut,\
labels=labels1))

age_income=welfare.dropna(subset="income")\
                  .groupby("age_group",as_index=False)\
                  .agg(mean_income = ("income","mean"))


agg.income

sns.barplot(data=age_income,x="age_group",y="mean_income")
plt.show()
plt.clf()



sex_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group","sex"],as_index=False)\
    .agg(mean_income = ("income","mean"))


np.version.version # np버전 확인

#판다스 데이터 프레임을 다룰 때, 변수의 타입이
#카테고리로 설정되어 있는 경우, groupby+agg 콤보 
#안먹힘. 그래서 object 타입으로 바꿔 준 후 수행

welfare["age_group"]=welfare["age_group"].astype("object")

sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group","sex"],as_index=False)\
    .agg(mean_income = ("income","mean"))


sex_age_income

sns.barplot(data=sex_age_income,
            x= "age_group",y="mean_income",hue="sex")
            
plt.show()
plt.clf()


# 연령대별 성별 상위 4% 수입 찾아보기

# dropna는 결측치 제거하기

from scipy.stats import norm
z_004=norm.ppf(0.96,loc=0,scale=1)

x.mean() +z_005 *6 /np.sqrt(16) 
x.mean() +z_005 *6 /np.sqrt(16) 


sex_age_income2 = \
    welfare.dropna(subset="income") \
    .groupby(["age_group","sex"],as_index=False)\
    .agg(mean_income = ("income",lambda x: np.quantile(x,q=0.96)))


sns.barplot(data=sex_age_income,
            x= "age_group",y="mean_income",hue="sex")
            
plt.show()
plt.clf()





#참고 income앞 []하나면 lsit [[]] 이렇게 두개면 dataframe
sex_income2 = welfare.dropna(subset = 'income') \
                    .groupby('sex', as_index = False)[['income']] \
                    .agg(['mean', 'std'])
                    
                    
#9-6 7.31

welfare["code_job"]
welfare["code_job"].value_counts()

list_job=pd.read_excel("./data/koweps/Koweps_Codebook_2019.xlsx",sheet_name="직종코드")
list_job.head()                    
                    
welfare=welfare.merge(list_job,how="left",on="code_job")

welfare.dropna(subset=["job","income"])[["income","job"]]

job_income = welfare.dropna(subset = ["job","income"]) \
                    .groupby('job',as_index=False) \
                    .agg(mean_income = ("income","mean"))
                    
job_income.head()                    


top10= job_income.sort_values("mean_income",ascending=False).head(10)                    

import matplotlib.pyplot as plt
plt.rcParams.update({"font.family" : "Malgun Gothic"})
sns.barplot(data = top10,y="job",x="mean_income",hue="job")
plt.show()
plt.clf()

job_income = welfare.dropna(subset = "job") \
                    .query("sex == 'male'") \
                    .groupby('job',as_index=False) \
                    .agg(n = ("job","count")) \
                    .sort_values("n",ascending =False) \
                    .head(10)
                    
    
# plt.tight_layout()


## 9-8
welfare["marriage_type"]
# query 조건에 맞는 행을 출력 groupby 기준에 맞게 묶고 value_counts(normalize =True) 원래는 숫자로 나왔지만 뒤에 normalize추가하면 비율로 변겅
df = welfare.query("marriage_type != 5") \
            .groupby('religion',as_index=False) \
            ["marriage_type"] \
            .value_counts(normalize =True) #핵심!
            
df        

rel_df = df.query('marriage_type ==1') \
            .assign(proportion=df["proportion"]*100) \
            .round(1)
