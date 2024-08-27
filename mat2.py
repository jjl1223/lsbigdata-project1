import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자능
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

from sklearn.linear_model import Lasso

model= Lasso(alpha=0.1) # lambda가 alpha표현됨
# alpha 즉 lambda가 커질 수록 계수가 0이되는게 증가한다
model.fit(train_x, train_y)

model.coef_

valid_df = df.loc[20:]
valid_df

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_x
valid_y = valid_df["y"]
valid_y

from sklearn.linear_model import Lasso

val_result=np.repeat(0.0, 100)
tr_result=np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train=sum((train_df["y"] - y_hat_train)**2)
    perf_val=sum((valid_df["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result


import seaborn as sns

df= pd.DataFrame({

 'lamda':np.arange(0,10,0.1),
 'tr': tr_result,
 'val' : val_result

})  

sns.scatterplot(data=df,x='lamda',y='tr')
sns.scatterplot(data=df,x='lamda',y='val')


#plt.xlim(0, 0.4)

val_result[0]
val_result[1]
np.min(val_result)

# alpha를 0.03로 선택! 0.03
# 0.03일때 vaild set의 val이 최소 이기 때문에
np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]




# 그려보기(mat_2_draw)

model= Lasso(alpha=0.03)
model.fit(train_x, train_y)

y_hat_val = model.predict(valid_x)
x=np.arange(-4,4,0.01)

show_df= pd.DataFrame({
    'x':x
})

for i in range(2, 21):
    show_df[f"x{i}"] = show_df["x"] ** i

y_show = model.predict(show_df)
plt.scatter(valid_x["x"],y_hat_val,color="blue") # 파란점 찍기
plt.plot(x,y_show,color="red") # 추정선 그리기
