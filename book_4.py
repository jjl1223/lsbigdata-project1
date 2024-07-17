#교재 4장
import pandas as pd
import numpy as np 
df= pd.DataFrame({
    'name' : ["김지훈","이유진","박동현","김민지"],
    'english' : [90,80,60,70],
    "math": [50,60,100,20]
}
)

df
df["name"]
type(df)
type(df["name"])
type(df[["name"]])
df["name"]
df[["name"]]
sum(df["english"])/4

dff=pd.DataFrame({
    "제품":['사과','딸기','수박'],
    "가격":[1800,1500,3000],
    "판매량":[24,38,13]
})




sum(dff["가격"])/3
sum(dff["판매량"])/3

import pandas as pd
import numpy as np    

df_exam=pd.read_excel("data/excel_exam.xlsx")
df_exam

df_exam["math"] /20
df_exam["english"] /20
df_exam["science"] /20


len(df_exam)
df_exam.shape
df_exam.size


df_exam=pd.read_excel("data/excel_exam.xlsx",
                     sheet_name="Sheet2"                            )

df_exam["total"] = df_exam["math"] + df_exam["english"] + df_exam["science"] #새로운 목록 추가
df_exam

df_exam["mean"] = df_exam["total"]/3
df_exam

type(df_exam["math"]>50) #논리연산시 series로 type변화
df_exam[df_exam["math"]>50] #math 50이상인 즉, true인값만 뽑혀서 나온다.

type(df_exam[df_exam["math"]>50]) #math 50이상인 즉, true인값만 뽑혀서 나온다.

df_exam[(df_exam["math"]>50) & (df_exam["english"]>50)]

mean_m=np.mean(df_exam["math"])
mean_e=np.mean(df_exam["english"])

df_exam[(df_exam["math"]>50) & (df_exam["english"]<mean_e)  ]


df_exam[df_exam["nclass"]==3][["math","english","science"]]

df_nc3 = df_exam[df_exam["nclass"]==3]

df_nc3[["math","english","science"]]
df_nc3[1:2]
df_nc3[1:2]

df_exam
df_exam[7:16]

df_exam[0::2]

df_exam.sort_values(["nclass","math"]) 
df_exam.sort_values(["nclass","math"],ascending =[True,False]) #True 오름차순 False 내림차순

a=np.array([4,2,5,3,6])
np.where(a>3,"Up","Down")
df_exam["updown"] = np.where(df_exam["math"]>50,"UP","Down")
df_exam
type(np.where(a>3,"Up","Down"))

np.where(a>3)
type(np.where(a>3))




