# 추정된 라쏘(lambda=0.03)모델을 사용해서,
# -4, 4까지 간격 0.01 x에 대하여 예측 값을 계산,
# 산점도에 valid set 그린 다음,
# -4, 4까지 예측값을 빨간 선으로 겹쳐서 그릴 것
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

np.random.seed(42)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df




for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]


valid_df = df.loc[20:]
valid_df

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_x
valid_y = valid_df["y"]
valid_y


model= Lasso(alpha=0.03)
model.fit(train_x, train_y)

y_hat_val = model.predict(valid_x)
x=np.linspace(-4,4,800)

show_df= pd.DataFrame({
    'x':x
})

for i in range(2, 21):
    show_df[f"x{i}"] = show_df["x"] ** i

y_show = model.predict(show_df)
plt.scatter(valid_x["x"],y_hat_val,color="blue") # 파란점 찍기
plt.plot(x,y_show,color="red") # 추정선 그리기\
plt.grid(True) # 격자 그리기

plt.legend()




import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

np.random.seed(42)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

# train 셋을 5개로 쪼개서
# vaild set과 train set을 5개로 만들기
# 각 세트에 대한 성을을 각 lambda값에 대응하여 구하기
# 성능평가 지표 5개를 평균내어 그래프 그리기


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

np.random.seed(42)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

vaild_df1 = df.loc[0:5]
vaild_df1

vaild_df2 = df.loc[6:11]
vaild_df2

vaild_df3 = df.loc[12:17]
vaild_df3

vaild_df4 = df.loc[18:23]
vaild_df4

vaild_df5 = df.loc[24:29]
vaild_df5

train_df1=df.drop(df.index[0:5])

train_df2=df.drop(df.index[6:11])

train_df3=df.drop(df.index[12:17])

train_df4=df.drop(df.index[18:23])

train_df5=df.drop(df.index[24:29])

def make_df(df):
    for i in range(2, 21):
        df[f"x{i}"] = df["x"] ** i
    return df

train_df1=make_df(train_df1)
train_df2=make_df(train_df2)
train_df3=make_df(train_df3)
train_df4=make_df(train_df4)
train_df5=make_df(train_df5)

vaild_df1=make_df(vaild_df1)
vaild_df2=make_df(vaild_df2)
vaild_df3=make_df(vaild_df3)
vaild_df4=make_df(vaild_df4)
vaild_df5=make_df(vaild_df5)


# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x1 = train_df1[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y1 = train_df1["y"]
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x2 = train_df2[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y2 = train_df2["y"]
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x3 = train_df3[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y3 = train_df3["y"]
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x4 = train_df4[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y4 = train_df4["y"]
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x5 = train_df5[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y5 = train_df5["y"]


# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x1 = vaild_df1[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_y1 = vaild_df1["y"]
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x2 = vaild_df2[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_y2 = vaild_df2["y"]
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x3 = vaild_df3[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_y3= vaild_df3["y"]
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x4 = vaild_df4[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_y4 = vaild_df4["y"]
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x5 = vaild_df5[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_y5 = vaild_df5["y"]

val_result=np.repeat(0.0, 100)
tr_result=np.repeat(0.0, 100)
a=[]

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x1)
    y_hat_val = model.predict(valid_x1)

    perf_train=sum((train_df1["y"] - y_hat_train)**2)
    perf_val=sum((vaild_df1["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result
val_result==val_result.min()
val_index=np.where(val_result==val_result.min())

val_index[0]*0.01

a.append(val_index[0][0]*0.01)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x2)
    y_hat_val = model.predict(valid_x2)

    perf_train=sum((train_df2["y"] - y_hat_train)**2)
    perf_val=sum((vaild_df2["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result
val_result==val_result.min()
val_index=np.where(val_result==val_result.min())

val_index[0]*0.01

a.append(val_index[0][0]*0.01)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x3)
    y_hat_val = model.predict(valid_x3)

    perf_train=sum((train_df3["y"] - y_hat_train)**2)
    perf_val=sum((vaild_df3["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result
val_result==val_result.min()
val_index=np.where(val_result==val_result.min())

val_index[0]*0.01

a.append(val_index[0][0]*0.01)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x4)
    y_hat_val = model.predict(valid_x4)

    perf_train=sum((train_df4["y"] - y_hat_train)**2)
    perf_val=sum((vaild_df4["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result
val_result==val_result.min()
val_index=np.where(val_result==val_result.min())

val_index[0]*0.01

a.append(val_index[0][0]*0.01)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x5)
    y_hat_val = model.predict(valid_x5)

    perf_train=sum((train_df5["y"] - y_hat_train)**2)
    perf_val=sum((vaild_df5["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result
val_result==val_result.min()
val_index=np.where(val_result==val_result.min())

val_index[0]*0.01

a.append(val_index[0][0]*0.01)