import pandas as pd
import numpy as np    

a = np.array([1,2,3,4,5])
b = np.array(["apple","banana"])
c = np.array([True,False])

print(type(a))
print(type(b))
print(type(c))

a[3]
type(a[2:])
a[1:4]


b= np.empty(3)
b
b[0]=1
b[1]=2
b[2]=3
b
vec1=np.arange(100)
vec1

vec2=np.arange(1,100)
vec2
#-100부터 0까지
vec3=np.arange(0,-100,-1) # 첫번째값:시작값 두번째값: 끝값 세번째값: 간격   시작값은 포함 끝값은 포함 x 
vec3=-np.arange(0,101)
vec3=np.arange(-100,0,1)
vec3

l_space1 = np.linspace(0,1,5)
#첫번째 값: 시작값 두번째 값: 끝값 세번째값: 시작값과 끝값사이에 세번째 값 개수만큼 일정한 간격으로 만들어줘
# repeat vs tile
reapea_1=np.repeat(l_space1,4)
reapea_1

tile_1= np.tile(l_space1,4)
tile_1

l_space1+l_space1

max(l_space1+l_space1)
sum(l_space1+l_space1)
# 35672 이하 홀수들의 합은?
vec1=np.arange(1,35672,2)
sum(vec1)
vec1.sum()


len(vec1)
vec1.shape


b = np.array([[1, 2, 3], [4, 5, 6]])
length= len(b)
shape= b.shape
size= b.size

c= np.array( [ [1,2] , [3,4] ] )
d= np.array( [ [1,2] , [3,4] ] )
c+d

e=(d ==3)
#35672 보다 작은수 중에서 7로 나눠서 나머지가 3인 숫자들의 개수
thi=np.arange(0,35672,1)
thi_1=thi%7
check=(thi_1==3)
sum(check)
#10보다 작은수 중에서 7로 나눠서 나머지가 3인 숫자들의 개수
thi=np.arange(0,10,1)%7
check=(thi==3)
sum(check)
sum(np.arange(0,10,1)%7 ==3)


