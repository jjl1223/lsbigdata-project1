import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
#성능 측정방법 https://white-joy.tistory.com/10
penguins = load_penguins()
penguins.head()

df=penguins.dropna()
df=df[["bill_length_mm","bill_depth_mm"]]

df=df.rename(columns={"bill_length_mm":'y',
          "bill_depth_mm" : 'x'})
#원래 MSE는?
np.mean((df["y"]-df["y"].mean())**2)

# x=15 기준으로 나눴을때, 데이터 포인트가 몇개 씩 나뉘나요?
n1=df.query('x>=15').shape[0]#1번그룹

n2=df.query('x<15').shape[0] #2번그룹

#1번그룹 얼마 예측?
#2번 그룹 얼마 예측?

df.query('x>=15')["y"].mean()

df.query('x<15').mean()[0]

#각 그룹 MSE는 얼마인가요?

mse1=np.mean((df.query('x>=15')["y"]-df.query('x>=15')["y"].mean())**2)
mse2=np.mean((df.query('x<15')["y"]-df.query('x<15')["y"].mean())**2)


# x=15의 MSE 가중평균은?
# (mse1+mse2)*0.5 가 아닌
(mse1 *n1+mse2*n2)/(n1+n2)
#x=20일때 MSE 가중평균은?




n1=df.query('x>=20').shape[0]#1번그룹

n2=df.query('x<20').shape[0] #2번그룹



df.query('x>=20')["y"].mean()

df.query('x<20').mean()[0]

mse1=np.mean((df.query('x>=20')["y"]-df.query('x>=20')["y"].mean())**2)
mse2=np.mean((df.query('x<20')["y"]-df.query('x<20')["y"].mean())**2)

(mse1 *n1+mse2*n2)/(n1+n2)

# 기준값 x를 넣으면 MSE값이 나오는 함수?
def mse(x) :
    n1 = df.query(f"x<{x}").shape[0] # 1번 그룹
    n2 = df.query(f"x>={x}").shape[0] # 2번 그룹
    y_hat1 = df.query(f"x<{x}")["y"].mean()
    y_hat2 = df.query(f"x>={x}")["y"].mean()
    mse1 = np.mean((df.query(f"x<{x}")["y"]-y_hat1)**2)
    mse2 = np.mean((df.query(f"x>={x}")["y"]-y_hat2)**2)
    return (mse1 * n1 + mse2 * n2) / (n1 + n2)

df["x"].min()
df["x"].max()

# 13~22 사이 값 중 
# minimize 사용해서 가장 작은 MSE가 나오는 x 찾아보세요

from scipy.optimize import minimize


# 초기 추정값
initial_guess = [0]

# 최소값 찾기
result = minimize(mse, initial_guess)
#안먹힘

# 13~22 사이 값 중 0.01간격으로 MSE계산해서
# 가장작은 x값 찾기
mse(13.0)
# x_values=np.linspace(start=df["x"].min()+0.01,stop=df["x"].max(),num=100)
np.arange(13,22,0.01)
x_values=np.arange(df["x"].min()+0.01,df["x"].max(),0.01)
x_values.shape

result = np.repeat(0.0,839)
result.shape
for i in range(839):
    result[i]=mse(x_values[i])

result
x_values[np.argmin(result)]
#

#16.4보다 작을때
df_min=df.query("x<16.42")
def mse_min(x) :
    n1 = df_min.query(f"x<{x}").shape[0] # 1번 그룹
    n2 = df_min.query(f"x>={x}").shape[0] # 2번 그룹
    y_hat1 = df_min.query(f"x<{x}")["y"].mean()
    y_hat2 = df_min.query(f"x>={x}")["y"].mean()
    mse1 = np.mean((df_min.query(f"x<{x}")["y"]-y_hat1)**2)
    mse2 = np.mean((df_min.query(f"x>={x}")["y"]-y_hat2)**2)
    return (mse1 * n1 + mse2 * n2) / (n1 + n2)
df_min["x"]
x_values=np.arange(df["x"].min()+0.01,16.4-0.1,0.01)
x_values.shape
result = np.repeat(0.0,x_values.shape[0])
for i in range(x_values.shape[0]):
    result[i]=mse_min(x_values[i])

result
x_values[np.argmin(result)] #14.1

#16.4보다 클때
df_max=df.query("x>=16.42")
def mse_max(x) :
    n1 = df_max.query(f"x<{x}").shape[0] # 1번 그룹
    n2 = df_max.query(f"x>={x}").shape[0] # 2번 그룹
    y_hat1 = df_max.query(f"x<{x}")["y"].mean()
    y_hat2 = df_max.query(f"x>={x}")["y"].mean()
    mse1 = np.mean((df_max.query(f"x<{x}")["y"]-y_hat1)**2)
    mse2 = np.mean((df_max.query(f"x>={x}")["y"]-y_hat2)**2)
    return (mse1 * n1 + mse2 * n2) / (n1 + n2)

x_values=np.arange(16.41,df["x"].max(),0.01)
x_values.shape
result = np.repeat(0.0,x_values.shape[0])
result.shape
for i in range(x_values.shape[0]):
    result[i]=mse_max(x_values[i])

result
x_values[np.argmin(result)] #16.41

df.plot(kind="scatter",x="x",y="y")
#14.02 16.42 19.4
plt.axvline(14.02, 0, 100, color='red', linestyle='--', linewidth=2)
plt.axvline(16.42, 0, 100, color='red', linestyle='--', linewidth=2)
plt.axvline(19.4, 0, 100, color='red', linestyle='--', linewidth=2)

threshold=[14.01,16.42,19.4]

df["group"]=np.digitize(df["x"],threshold)
df.groupby("group").mean()["y"]

y_mean=df.groupby("group").mean()["y"]
k1=np.linspace(13, 14.01, 100)
k2=np.linspace(14.01, 16.42, 100)
k3=np.linspace(16.42, 19.4, 100)
k4=np.linspace(19.4, 22, 100)
plt.plot(k1, np.repeat(y_mean[0],100), color="red")
plt.plot(k2, np.repeat(y_mean[1],100), color="red")
plt.plot(k3, np.repeat(y_mean[2],100), color="red")
plt.plot(k4, np.repeat(y_mean[3],100), color="red")