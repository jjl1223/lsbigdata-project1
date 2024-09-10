# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘라스틱 넷 & 디시젼트리 회귀모델 사용
# 모든 변수 자유롭게 사용!
# 종속변수 : bill_length_mm
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from palmerpenguins import load_penguins
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
penguins = load_penguins()
penguins.head()

penguins.isna().sum().sum()

# 각 숫자변수는 평균채우기
# 숫자형 채우기
quantitative = penguins.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    penguins[col].fillna(penguins[col].mean(), inplace=True)
penguins[quant_selected].isna().sum()

# 각 문자변수는 unknow 채우기
# 범주형 채우기
quantitative = penguins.select_dtypes(include = [object])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    penguins[col].fillna("unknown", inplace=True)
penguins[quant_selected].isna().sum()





df=penguins
df_x=df.drop("bill_length_mm", axis=1)
df_y=df["bill_length_mm"]
df_basic = df[["bill_length_mm", "bill_depth_mm"]]
df_x = pd.get_dummies(
    df_x,
    columns= df_x.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df_x
df_x = pd.get_dummies(
    df_x,
    columns= df_x.select_dtypes(include=[int,float]).columns,
    drop_first=True
    )
df_x


param_grid={
    "alpha" : np.arange(3,17,1),
    "l1_ratio" : np.arange(0,7,1)

}
model= ElasticNet()
grid_search= GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",# 그냥 mean_squared_error다 이름따지면 문제가 많다
    cv=5 # 교차모델을 5개 만들어서 해라
         # test train 나누는걸  5개만들어라
)
grid_search.fit(df_x,df_y)

grid_search.best_params_ # 최고모델 알파 l1_ratio값 알려줌
grid_search.cv_results_ # 모델 결과
grid_search.best_score_ # 모델점수(성능지표)
best_model=grid_search.best_estimator_ #이렇게 하면 바로 최고모델을 집어넣을 수 있다

# 디시전 트리 회귀 모델 생성 및 학습
#min_samples_split : 특정노드에 남아있는 최소개수
#min_samples_leaf : 몇개남으면 분할을 멈출지 결정
#max_features : 분할할때 몇개의 열(변수)로 구별할것인가 
#"max_depth" : 최대 깊이 
#splitter : spliter 옵션 중 best는 MSE를 최소로 하는 기준을 골라주고, 
# Random은 MSE를 고려하지 않고 이녀석이 골라준 기준값



param_grid={
    "max_depth" : np.arange(0,10,1),
    "min_samples_split" : np.arange(0,10,1),
    "min_samples_leaf" : np.arange(0,10,1)
}

model = DecisionTreeRegressor(random_state=42) 
grid_search= GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",# 그냥 mean_squared_error다 이름따지면 문제가 많다
    cv=5 # 교차모델을 5개 만들어서 해라
         # test train 나누는걸  5개만들어라
)
grid_search.fit(df_x,df_y)

grid_search.best_params_



model = DecisionTreeRegressor(random_state=42,
                              max_depth=2,
                              min_samples_split=22)
model.fit(df_x,df_y)

from sklearn import tree
tree.plot_tree(model)


