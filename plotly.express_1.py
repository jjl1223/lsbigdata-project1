#pip install palmerpenguins
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
#데이터 로드
penguins = load_penguins()
penguins.head()
penguins["species"].unique
penguins.columns

# 범례 이름 한글로 변경
species_map = {
    "Adelie": "아델리",
    "Chinstrap": "턱끈",
    "Gentoo": "젠투"
}
penguins['species_ko'] = penguins['species'].map(species_map)

#산포 생성

fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color = "species",
    size_max=10, #기본점 크기 설정
    trendline="ols"# p.134 회귀직선 그려주기
)


fig.update_layout(
    #font는 색지정,size는 크기 지정 아마도 글씨체 지정도 가능할듯
    title=dict(text="팔머펭귄 부리길이 vs 부리깊이",font=dict(color="white",size=24)),
    paper_bgcolor = "black",#목차 제목 색
    plot_bgcolor = "black",#그래프 내부색
    xaxis = dict(
        title=dict(text="뿌리 길이 vs 깊이", font=dict(color="white")),
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
        ),
     yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(font=dict(color="white"))
)

# 점 크기 업데이트
fig.update_traces(marker=dict(size=12))  # 점 크기 설정




fig.show()

# 회귀직선 그리기
#pip install scikit-learn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
penguins=penguins.dropna()
x=penguins[["bill_length_mm"]]
y=penguins["bill_depth_mm"]

model.fit(x, y)
linear_fit=model.predict(x)
model.coef_
model.intercept_
#전체 회귀직선으로 보면 줄어드는것처럼 보이지만 각 집단별로 보면 늘어난다 이거를 심슨의 역설이라고 한다

fig.add_trace(
    go.Scatter(
        mode="lines",
        x=penguins["bill_length_mm"], y=linear_fit,
        name="선형회귀직선",
        line=dict(dash="dot", color="white")
    )
)
fig.show()

# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환(T,F로 변환) 즉 문자열을 숫자로 변환가능
penguins_dummies = pd.get_dummies(penguins, columns=['species'], drop_first=True)

#첫번째행을 버려도 되는 이유는 FF일때 당연히 첫번째꺼이길것이기 때문에 데이터양을 줄이기 위해서 버린다

penguins_dummies.columns
penguins_dummies.iloc[:,-3:]

#x와 y설정
x= penguins_dummies[["bill_length_mm","species_Chinstrap",'species_Gentoo']]
y=penguins_dummies["bill_depth_mm"]


model = LinearRegression()
model.fit(x, y)

model.coef_
model.intercept_

regline_y=model.predict(x)

import matplotlib.pyplot as plt
plt.scatter(x["bill_length_mm"],y,color="black",s=1)
hue=penguins(["species"])
plt.scatter(x["bill_length_mm"],regline_y,s=1)

# y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56
# penguins
# species    island  bill_length_mm  ...  body_mass_g     sex  year
# Adelie     Torgersen            39.5  ...       3800.0  female  2007
# Chinstrap  Torgersen            40.5  ...       3800.0  female  2007
# Gentoo     Torgersen            40.5  ...       3800.0  female  2007
# x1, x2, x3
# 39.5, 0, 0
# 40.5, 1, 0
# y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56







  
