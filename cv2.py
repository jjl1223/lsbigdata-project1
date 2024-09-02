import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

# 데이터를 DataFrame으로 변환하고 다항 특징 추가
x_vars = np.char.add('x', np.arange(1, 21).astype(str))
X = pd.DataFrame(x, columns=['x'])
# 다항함수를 만들어주는데 include_bias는 x의0승을 안만들어준다고 생각하면된다
poly = PolynomialFeatures(degree=20, include_bias=False)

X_poly = poly.fit_transform(X) #
X_poly=pd.DataFrame(
    data=X_poly,
    columns=x_vars
)

# 교차 검증 설정
kf = KFold(n_splits=3, shuffle=True, random_state=2024) #셔플한 값 만들기

# 알파 값 설정
alpha_values = np.arange(0, 10, 0.01)

# 각 알파 값에 대한 교차 검증 점수 저장
mean_scores = []

for alpha in alpha_values:
    lasso = Lasso(alpha=alpha, max_iter=5000) # 정확하게 구하라
    #lasso 모델 받고 X_poly x값 받고 y로 y값 받기 (train date 다 집어넣고) cv값으로 받아온 kf
    #값으로 어떤 것을 train으로 쓸지 어떤것을 valid(test)로 쓸지 결정해서 알아서 
    # train해주고 valid set으로 점수를 측정해서 해줌
    # scoring = 사각형 넓이로 성능을 측정하는데 neg를 넣으면 낮은게 좋다 오차의 제곱에 음수를
    # 붙인게 아니다 함수가 이상하다 그냥 받아들여라
    scores = cross_val_score(lasso, X_poly, y, cv=kf, scoring='neg_mean_squared_error')
    # valid set에 대한 score가 각각 3개가 나오므로 그거에 대한 평균을 내준다
    mean_scores.append(np.mean(scores))
  

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)