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
