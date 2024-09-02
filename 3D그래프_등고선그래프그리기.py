import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-1, 7, 400)
y = np.linspace(-1, 7, 400)
x, y = np.meshgrid(x, y)
# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3
# 그래프를 그리기 위한 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 표면 그래프를 그립니다.
ax.plot_surface(x, y, z, cmap='viridis')
# 레이블 및 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Graph of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()

# ==========================
# 등고선 그래프

import numpy as np
import matplotlib.pyplot as plt

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-1, 7, 400)
y = np.linspace(-1, 7, 400)
x, y = np.meshgrid(x, y) # 점 400*400개 전부 만들기

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

#특정 점 (9,2)에 파란색 점을 표시
plt.scatter(9,2,color="red",s=50)

x=9; y=2
lstep=0.1
for i in range(10):
    x,y= np.array([x,y]) - lstep  * np.array([2*x-6,2*y-8])
    plt.scatter(x,y,color="red",s=50)
x
y





# 축 레이블 및 타이틀 설정
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()




z=(1-(x+y))**2 + (4-(x+2*y))**2 + (1.5-(x+3*y))**2 + (5-(x+4*y))**2
f_x=-23+8*x+20*y
f_y=-67+20*x+60*y
x=10
y=10
lstep=np.arange(100,1,-1)*0.01
for i in range(100000):
    f_x=-23+8*x+20*y
    f_y=-67+20*x+60*y
    x=x-0.01*f_x
    y=y-0.01*f_y
x    
y

# Q. 다음을 최소로 만드는 베타 벡터
# f(beta0, beta1) = (1-(beta0+beta1))^2 +
#                   (4-(beta0+2*beta1))^2 +
#                   (1.5-(beta0+3*beta1))^2 +
#                   (5-(beta0+4*beta1))^2
# 초기값 : (10,10)
# delta : 0.01

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# beta0, beta1의 값을 정의합니다 
beta0 = np.linspace(-20, 20, 100)
beta1 = np.linspace(-20, 20, 100)
beta0, beta1 = np.meshgrid(beta0, beta1)

# 함수 f(beta0, beta1)를 계산합니다.
z = (1-(beta0+beta1))**2 + (4-(beta0+2*beta1))**2 + (1.5-(beta0+3*beta1))**2 + (5-(beta0+4*beta1))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(beta0, beta1, z, levels=100)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점 (10,10)에 빨간색 점을 표시
plt.scatter(10, 10, color="red", s=50)

# x(100), y(100) 구하는 방법
beta0 = 10; beta1 = 10
lstep = 0.01
for i in range(100) : 
    (beta0, beta1) = np.array([beta0, beta1]) - lstep * np.array([8*beta0 + 20*beta1 -23, 20*beta0 + 60*beta1 -67])
    plt.scatter(beta0,beta1,color="red", s=50)
print(beta0,beta1)

# 그래프 표시
plt.show()