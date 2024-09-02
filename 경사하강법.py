# y=(x-2)^2 + 1 그래프 그리기
# 점을 직선으로 이어서 표현
# 최적화 PPT파일 참고
import pandas as pd
import seaborn as sns    
import numpy as np
import matplotlib.pyplot as plt

k=6
x = np.linspace(-4, 8, 100)
y = (x - 2)**2 + 1
# plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")
plt.xlim(-4, 8)
plt.ylim(0, 15)

# f'(x)=2x-4
# k=4의 기울기
l_slope=2*k - 4
f_k=(k-2)**2 + 1
l_intercept=f_k - l_slope * k

# y=slope*x+intercept 그래프
line_y=l_slope*x + l_intercept
plt.plot(x, line_y, color="red")


# 미분의 의미: 0이 되는게 최대값 or 최솟값
# 최솟값을 찾기위해서는 미분값이 양수인경우 왼쪽으로 음수인경우는 오른쪽으로 가면 
# 최솟값을 찾을 수있다. 미분값이 0 일때는 최대값일떄는 저거 반대가 되겠지만
# 우리 목표는 최솟값을 찾는것이기 떄문에 저거에는 적용을 하지 않는다
# minimize 함수가 위의원리를 쓰는데 왼쪽이나 오른쪽으로 가는 보폭을 매우 작게 해서
# 이동하면 찾을 수 있다. 매우 작은 보폭을 스텝이라고 한다 이때 기울기의 절대값
# 이 클수록 최솟값에서 멀리떨어져 있다는 것이기 때문에 이를 반영하기 위해서
# step * f'(x)를 해서 이동해서 찾는 속도를 빠르게 한다
# 이를 경사하강법이라고 한다.

# 경사하강법으로 최솟값 찾아보기
# 양수일떄 왼쪽 음수일떄 오른쪽이므로 f_X값에 -를 곱해주고 가중치 곱해주기

f_x=2*x
#초기값 10 델타 0.9
x=10
lstep=0.9
np.arange(100,1,-1)
np.arange(1,100,2)

lstep=np.arange(100,1,-1)*0.01
range(10000)
for i in range(10000):
    f_x=2*x
    x=x-lstep*f_x
x    


f_x= 2*x-6
f_y= 2*y-8
x=9
y=2
lstep=np.arange(100,1,-1)*0.01
range(10000)
for i in range(10):
    f_x= 2*x-6
    f_y= 2*y-8
    x=x-lstep*f_x
    y=y-lstep*f_y
x    
y
#!pip install sympy
from sympy import Derivative, symbols
z=(1-(x+y))**2 + (4-(x+2*y))**2 + (1.5-(x+3*y))**2 + (5-(x+4*y))**2
f_x=-23+8*x+20*y
f_y=-67+20*x+60*y
x=10
y=10
lstep=np.arange(100,1,-1)*0.01
for i in range(10000000):
    f_x=-23+8*x+20*y
    f_y=-67+20*x+60*y
    x=x-0.01*f_x
    y=y-0.01*f_y
x    
y

