import pandas as pd
import numpy as np    

df=pd.DataFrame({"sex":["M","F",np.nan,"M","F"],
                "score": [5,4,3,4,np.nan]                                })
                
df

df["score"] +1
pd.isna(df)

pd.isna(df).sum()

# 결측치 제거하기
df.dropna() #모든 변수 결측치 제거
df.dropna(subset = "score") # score 변수에서 결측치 제거

df.dropna(subset = ["score","sex"]) # 여러변수 결측치 제거
df[df["score"] ==4.0]

exam=pd.read_csv("data/exam.csv")
# 데이터 프레임 location을 사용한 인덱싱
#exam.loc[행 인덱스, 열 인덱스]
exam.loc[[0],["id","nclass"]]
exam.iloc[0:2,0:4] =3
exam.iloc[[2,7,4],["math"]] = np.nan
exam.iloc[[2,7,4],2] = np.nan
exam
exam.iloc[[2,7,4],2] = 3
exam
exam.loc[[2,7,14],]

type(exam.iloc[[2,7,4],2])
type(exam.iloc[0:2,0:4])
exam.iloc[1,]

df[df["score"]==3.0]["score"]
type(df[df["score"]==3.0]["score"])

#수학점수 50점 이하인 학생들 점수 50으로 상향 조정
exam

exam.loc[exam["math"]<=50,"math"] =50
#영어점수 90점 이상 90점으로 하향 조정(iloc 사용) 무조건 숫자로 조회

exam.iloc[exam["english"]>=90,3] #실행안됨

exam.iloc[np.array(exam["english"]>=90),3] #실행됨
exam.iloc[exam[exam["english"]>=90].index,3] #np.where도 듀플이라서 [0] 사용해서 꺼내오면 동작
exam.iloc[np.where([exam["english"]>=90])[0],3] # index백터도 작동

exam.iloc[np.array(exam["english"]>=90),3]=90
exam.iloc[exam[exam["english"]>=90].index,3]=90
exam.iloc[np.where([exam["english"]>=90])[0],3]=90

#math 점수 50 이하 "-" 변경
exam

exam.loc[exam["math"]<=50,"math"] ='-' # True인값에만 값을 넣는다 false에는 넣지 않음
exam
exam.loc[exam["math"]== '-' ,"math"] = exam[exam["math"] != "-"]["math"].mean()

exam.loc[(exam["math"])]
exam.query('math != ["-"]')['math'].mean()
exam.query('math not in ["-"]')['math'].mean()



np.mean(exam["math"] != "-","math") 

np.nan

exam[exam["math"] != "-"]["math"].mean()


df loc[df["score"]]

#6
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam["math"] = exam["math"].replace("-",math_mean)

exam




