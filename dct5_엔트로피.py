import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

#펭귄 분류 문제
#y: 펭귄의 종류
#x1: bill_length_mm(부리 길이)
#x2: bill_depth_mm(부리 깊이)

df=penguins.dropna()
df=df[["species","bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={
    'bill_length_mm': 'x1',
    'bill_depth_mm': 'x2',
    "species" : "y"})
df

#x1,x2 산점도를 그리되, 점 색깔은 펭귄 종별 다르게 그리기

import  matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(data=df,x="x1",y="x2",hue="y")

plt.axvline(x=45)

#Q. 나누기 전 현재의 엔트로피?
#Q. 45로 나눴을때, 엔트로피 평균은 얼마인가요?
# 입력값이 벡터 -> 엔트로피!

p_i=df['y'].value_counts() / len(df['y'])
entropy_curr=-sum(p_i * np.log2(p_i))

n1=df.query("x1>=45").shape[0]
n2=df.query("x1<45").shape[0]


# x1=45 기준으로 나눈 후 ,평균 엔트로피

a=df.query("x1>=45") #1번 그룹
b=df.query("x1<45") # 2번 그룹
# 어떤 종류로 예측할지
y_hat1=df.query("x1>=45")["y"].mode()
y_hat2=df.query("x1<45")["y"].mode()

# 각 그룹 엔트로피는 얼마인가?
p_1=df.query("x1>=45")["y"].value_counts()/len(df.query("x1>=45")["y"])
entropy_curr1=-sum(p_1*np.log2(p_1))

p_2=df.query("x1<45")["y"].value_counts()/len(df.query("x1<45")["y"])
entropy_curr2=-sum(p_2*np.log2(p_2))

#가중평균 반영

entropy_x145=(n1*entropy_curr1+n2*entropy_curr2)/(n1+n2)
entropy_x145

# 엔트로피 구하는 함수
# x1 기준으로 최적 기준값은 얼마인가?
# 기준값 x를 넣으면 MSE값이 나오는 함수는?
def my_entropy(x):
    n1=df.query(f"x1 < {x}").shape[0]  # 1번 그룹
    n2=df.query(f"x1 >= {x}").shape[0] # 2번 그룹
    p_1=df.query(f"x1 < {x}")["y"].value_counts()/len(df.query(f"x1 < {x}")["y"])
    entropy_curr1=-sum(p_1*np.log2(p_1))
    p_2=df.query(f"x1 >= {x}")["y"].value_counts()/len(df.query(f"x1 >= {x}")["y"])
    entropy_curr2=-sum(p_2*np.log2(p_2))
    return float(n1*entropy_curr1+n2*entropy_curr2)/(n1+n2)
from scipy.optimize import minimize


# 초기 추정값
initial_guess = [0]

# 최소값 찾기
result = minimize(my_entropy, initial_guess)
# 또 안먹힘

n1=df.query(f"x1 < {42.30999999999797}").shape[0]
n2=df.query(f"x1 >= {42.30999999999797}").shape[0]
x_values=np.arange(df["x1"].min(),df["x1"].max(),0.01)
x_values.shape
result = np.repeat(0.0,x_values.shape[0])
for i in range(x_values.shape[0]):
    result[i]=my_entropy(x_values[i])

result
x_values[np.argmin(result)] 

