import numpy as np

# 백터 백터 내적
a=np.arange(1,4)
a
b= np.array([3,6,9])
b
a.dot(b)

#
a=np.array([1,2,3,4]).reshape((2,2),order="F") # order="F" 세로로 쌓기
a

b=np.array([5,6]).reshape(2,1)
b

a.dot(b)

a @ b
#Q1

a=np.array([1,2,1,0,2,3]).reshape((2,3),order="c")

b=np.array([1,0,-1,1,2,3]).reshape((3,2))

a @b

#Q2

np.eye(3) # 단위행렬 만들기
a=np.array([3,5,7,
            2,4,9,
            3,1,0]).reshape((3,3),order="c")

a @ np.eye(3)

# transpose
a.transpose()
b=a[:,0:2]
b.transpose()

# 행렬로 데이터를 계산하는이유?
# 데이터를 축약해서 받아와서 계산할수 있다
# ex) 펭귄 부리깊이와 날개길이를 회귀분석해서 회귀게수(coef_)와 intercept
# 를 받아왔을때 왼쪽행렬에 coef와 intercept를 행으로 하는 행렬과 부리깊이와
# 날개길이의 개수와 같은 열로 배치하면 수월하게 원하는 변수를 예측하는 것을 구할 
# 수 있다. 


#회귀계수 구하기 벡터자료 참고하기(강의 자료에 있음)

# 회귀분석 데이터행렬
x=np.array([13,15,
            12,14,
            10,11,
            5,6]).reshape(4,2)
x
vec1=np.repeat(1,4).reshape(4,1) # 절편 즉 intercept에 1을 곱하는 꼴을 만들기 위해서 만들고
matX=np.hstack((vec1,x)) # 절편 곱해주는것을 만든다

beta_vec=np.array([2,0,1]).reshape(3,1)
beta_vec

matX@beta_vec

y=np.array([20,19,20,12]).reshape(4,1)

(y-matX @ beta_vec).transpose() @ (y-matX @ beta_vec)

beta_vec=np.array([2,0,1]).reshape(3,1)

#역행렬

c=np.array([1,5,3,4]).reshape(2,2)
c_inv=np.array([4,-5,-3,1]).reshape(2,2)*(1/-11)

c@c_inv

# 3 by 3 역행렬
a=np.array([-4,-6,2,
            5,-1,3,
            -2,4,-3]).reshape(3,3)

a_inv=np.linalg.inv(a)
a_inv
a @ a_inv

np.round(a @ a_inv)

# 세로값들 즉 각 열들이 선형독립일때만 역행렬을 구할 수 있다
# 역행렬이 없는 행렬을 특이행렬이라고 한다.

b=np.array([1,2,3,
            2,4,5,
            3,6,7]).reshape(3,3)

b_inv=np.linalg.inv(b) # 에러남

np.linalg.det(b) #행렬식이 항상 0

#베타구하기
matX
y

XtX_inv=np.linalg.inv((matX.transpose()@matX))

Xty=matX.transpose() @ y
beta_hat=XtX_inv @ Xty
beta_hat


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(matX[:,1:], y)

model.coef_
model.intercept_

reg_line=model.predict([13,21,22])
# minimize로 베타 구하기
import numpy as np
from scipy.optimize import minimize

def line_perform(beta):
   beta=np.array(beta).reshape(3,1)
   a=(y-matX @ beta)
   return (a.transpose() @ a) # 정사각형 넓이합(벡터끼리 곱하기 위해 transpose해서 구하기)

line_perform([6, 1,3])

# 초기 추정값
initial_guess = [0, 1, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# minimize로 라쏘 베타 구하기
import numpy as np
from scipy.optimize import minimize

def line_perform_lasso(beta):
   beta=np.array(beta).reshape(3,1)
   a=(y-matX @ beta)
   return (a.transpose() @ a) +30*np.abs(beta[1:]).sum()
# 30이 람다이다
line_perform_lasso([8.55, 5.96,-4.38])
line_perform_lasso([8.14,0.96,0])
# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


[8.55,5.96,-4.38] #람다 0

[8.14,0.96,0] #람다 3
#예측식: y_hat = 8.14+0.96*x1 + 0 *x2

[17.74,0,0] #람다 500
#예측식: y_hat = 17.74+0*x1 + 0 *x2

#람다를 잘 설정하면 괜찮은 변수를 선택할 수 있다.
#라쏘베타를 활용해서 필수적인 변수를 구할 수 있다.
#저 값이 0이면 사실상 영향이 없는 변수이기때문에 제외할 수 있다

# x변수가 추가되면 trainX에서는 성능이 항상 좋아진다
# x변수가 추가되면 validX에서는 좋아졌다가 나빠짐(오버피팅)
#어느 순간 X변수 추가하는 것을 멈춰야 한다

#람다 0 부터 시작: 내가 가진 모든 변수를 넣겠다
# 점점 람다를 증가: 변수가 하나씩 빠지는 효과
# validX에서 가장 성능이 좋은 람다를 선택!
# 변수가 선택됨을 의미.

# x의 칼럼에 선형 종속인 애들이 있다: 다중공선성이 존재한다.
# 다중공선성이 많으면 함수가 이쁘게 1차2차함수로 생기는게 아니라
# 미분한 값이 0이 나오는값이 많아지는 그래프가 된다
# 이때 최솟값이 초기값에 의해서 많이 달라지는데 이러한 문제때문에
# 최솟값을 구하기 쉽지않다
# 




# minimize로 릿지 베타 구하기
import numpy as np
from scipy.optimize import minimize

def line_perform_ridge(beta):
   beta=np.array(beta).reshape(3,1)
   a=(y-matX @ beta)
   return (a.transpose() @ a) +3*(beta**2).sum()

line_perform_ridge([8.5, 5.96,-4.38])

# 초기 추정값
initial_guess = [0, 1, 0]

# 최소값 찾기
result = minimize(line_perform_ridge, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)