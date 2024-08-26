import numpy as np
import matplotlib.pyplot as plt

a=4
b=-12
c=0.7
d=12
e=32
x = np.linspace(-4,4,100)
y= a*x **4+ b*x**3+c*x**2+d*x+e

plt.plot(x,y,color="black")

#데이터 만들기
from scipy.stats import norm
from scipy.stats import uniform

norm.rvs(size=100,loc=0,scale=0.3)
# 파란 점들
x=uniform.rvs(size=20,loc=-4,scale=8)
y=np.sin(x) + norm.rvs(size=20,loc=0,scale=0.3)
# 검정 곡선
k=np.linspace(-4,4,200)
sin_y=np.sin(k) 

plt.plot(k,sin_y,color='black')
plt.scatter(x,y,color='blue')


# train,test 데이터 만들기
np.random.seed(42)
x=uniform.rvs(size=30,loc=-4,scale=8)
y=np.sin(x) + norm.rvs(size=30,loc=0,scale=0.3)

import pandas as pd
df=pd.DataFrame({

    "x":x,"y":y
})
df

train_df=df.iloc[:20]
train_df

test_df=df.loc[20:]
test_df
train_df.iloc[:,0]
plt.scatter(train_df["x"],train_df["y"],color="blue")

from sklearn.linear_model import LinearRegression

model = LinearRegression()

# 모델 학습
x=train_df[['x']]
y=train_df['y']
model.fit(x, y)

model.coef_
model.intercept_

reg_line=model.predict(x)

plt.plot(x,reg_line,color="red")
plt.scatter(train_df["x"],train_df["y"],color="blue")

#2차 곡선 회귀 시작

train_df["x_2"] = train_df["x"] **2


x=train_df[["x","x_2"]]
y=train_df['y']

model.fit(x, y)

model.coef_
model.intercept_

# 회귀직선 그려보기 위해서 값만들기 
# x값이 크기 순서대로 나온게 아니기때문에 선이 뒤로갔다 앞으로 갔다해서 이상하게 나온다
# x를 x.sort_values("x")를 통해서 그리면 그럭저럭 이쁘게 된다
# 단, 이렇게 그리는것 만큼 매끄럽지 않다
k= np.linspace(-4,4,200)
df_k = pd.DataFrame({
    "x":k,"x_2":k**2
})
df_k
reg_line = model.predict(df_k)
# 그래프 그리기 회귀직선이 1차함수가 아니라 살짝 곡선이 되는 것을 확인할 수 있다
plt.plot(k,reg_line,color="red")
plt.scatter(train_df["x"],train_df["y"],color="blue")

#3차 곡선 회귀

train_df["x_3"] = train_df["x"] **3


x=train_df[["x","x_2","x_3"]]
y=train_df['y']

model.fit(x, y)

model.coef_
model.intercept_

# 회귀직선 그려보기 위해서 값만들기
k= np.linspace(-4,4,200)
df_k = pd.DataFrame({
    "x":k,"x_2":k**2,"x_3":k**3
})
df_k
reg_line = model.predict(df_k)
# 그래프 그리기 회귀직선이 2차회귀에 비해 매우 잘따라간다
plt.plot(k,reg_line,color="red")
plt.scatter(train_df["x"],train_df["y"],color="blue")

#4차 곡선 회귀

train_df["x_4"] = train_df["x"] **4


x=train_df[["x","x_2","x_3","x_4"]]
y=train_df['y']

model.fit(x, y)

model.coef_
model.intercept_

# 회귀직선 그려보기 위해서 값만들기
k= np.linspace(-4,4,200)
df_k = pd.DataFrame({
    "x":k,"x_2":k**2,"x_3":k**3,"x_4":k**4
})
df_k
reg_line = model.predict(df_k)
# 그래프 그리기 회귀직선이 3차회귀와 매우 비슷하다
plt.plot(k,reg_line,color="red")
plt.scatter(train_df["x"],train_df["y"],color="blue")

#9차 곡선 회귀

train_df["x_5"] = train_df["x"] **5
train_df["x_6"] = train_df["x"] **6
train_df["x_7"] = train_df["x"] **7
train_df["x_8"] = train_df["x"] **8
train_df["x_9"] = train_df["x"] **9


x=train_df[["x","x_2","x_3","x_4","x_5","x_6","x_7","x_8","x_9"]]
y=train_df['y']

model.fit(x, y)

model.coef_
model.intercept_

# 회귀직선 그려보기 위해서 값만들기
k= np.linspace(-4,4,200)
df_k = pd.DataFrame({
    "x":k,"x_2":k**2,"x_3":k**3,"x_4":k**4,"x_5":k**5,"x_6":k**6
    ,"x_7":k**7,"x_8":k**8,"x_9":k**9
})
df_k
reg_line = model.predict(df_k)
# 그래프 그리기 회귀직선이 정답이 sin graph와 다르게 너무 overfit되는것을 알 수있다
plt.plot(k,reg_line,color="red")
plt.scatter(train_df["x"],train_df["y"],color="blue")



#14차 곡선 회귀

train_df["x_10"] = train_df["x"] **10
train_df["x_11"] = train_df["x"] **11
train_df["x_12"] = train_df["x"] **12
train_df["x_13"] = train_df["x"] **13
train_df["x_14"] = train_df["x"] **14


x=train_df[["x","x_2","x_3","x_4","x_5","x_6","x_7","x_8","x_9","x_10",
            "x_11","x_12","x_13","x_14"]]
y=train_df['y']

model.fit(x, y)

model.coef_
model.intercept_

# 회귀직선 그려보기 위해서 값만들기
k= np.linspace(-4,4,200)
df_k = pd.DataFrame({
    "x":k,"x_2":k**2,"x_3":k**3,"x_4":k**4,"x_5":k**5,"x_6":k**6
    ,"x_7":k**7,"x_8":k**8,"x_9":k**9,"x_10":k**10,"x_11":k**11,
    "x_12":k**12,"x_13":k**13,"x_14":k**14
})
df_k
reg_line = model.predict(df_k)
# 그래프 그리기 회귀직선이 정답이 sin graph와 다르게 너무 overfit되는것을 알 수있다
plt.plot(k,reg_line,color="red")
plt.scatter(train_df["x"],train_df["y"],color="blue")

# 9차 모델성능확인 다른방법
test_df["x2"] = test_df["x"]**2
test_df["x3"] = test_df["x"]**3
test_df["x4"] = test_df["x"]**4
test_df["x5"] = test_df["x"]**5
test_df["x6"] = test_df["x"]**6
test_df["x7"] = test_df["x"]**7
test_df["x8"] = test_df["x"]**8
test_df["x9"] = test_df["x"]**9

x=test_df[["x", "x2", "x3", "x4", "x5",
           "x6", "x7", "x8", "x9"]]
y_hat=model.predict(x)
# test_df:실제값 y_hat: 모델에서 나온값
sum((test_df["y"] - y_hat)**2)

# 9차 모델성능확인 른방법

np.linspace(2,11,10)
#f string
for a in np.linspace(2,9,8):
    test_df[f"x_{a:.0f}"]=test_df["x"] ** a

x=test_df[["x","x_2","x_3","x_4","x_5","x_6","x_7","x_8","x_9"]]
y=test_df['y']


y_hat=model.predict(x)

# test_df:실제값 y_hat: 모델에서 나온값
sum((test_df["y"] - y_hat)**2)








# 20차 모델 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression




# 20차 모델 성능을 알아보자
np.random.seed(42)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "x" : x , "y" : y
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
y = train_df["y"]

model=LinearRegression()
model.fit(x,y)

test_df = df.loc[20:]
test_df

for i in range(2, 21):
    test_df[f"x{i}"] = test_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
x = test_df[["x"] + [f"x{i}" for i in range(2, 21)]] #리스트 컴프리헨션

y_hat = model.predict(x)

# 모델 성능
sum((test_df["y"] - y_hat)**2)