#lec4장 이어서
n  a= np.array([1.0,2.0,3.0])
b=2.0
a*b

a.shape
b.shape

import numpy as np

matrix = np.array( [ [0.0,0.0,0.0],
                  [10.0,10.0,10.0],
                  [20.0,20.0,20.0],
                  [30.0,30.0,30.0]] ) # []하나가 차원을 증가시키는것

matrix.shape

vector = np.array([1.0,2.0,3.0])
vector.shape

result = matrix + vector
print("브로드캐스팅 결과:\n", result)



vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(3,)

a= np.array([1.0,2.0,3.0],[3,4,5])
a[a >= 10] = 10


np.random.seed(2024)
a=np.random.randint(1,26346,1000)
a

#처음으로 5000보다 큰 숫자가 나오는 위치는?

result=a[a>=5000]
result1_=np.where(a>=5000)


a[a>5000][4] #5000보다 큰 5번째에 있는 숫자
np.where(a>=20000)[0][0]



result1_=np.where(a>=5000)
result1_
type(result1_)
result1_[0]
type(result1_[0])
result1_[0][12]


#처음으로 24000보다 큰 숫자 나왔을때,
#숫자 위치와 그 숫자는 무엇인가요?
np.random.seed(2024)
a=np.random.randint(1,26346,1000)
a
b=np.where(a>=24000)
my_index=b[0][0]
a[my_index]
my_index+3

#처음으로 10000보다 큰 숫자들 중 50번째로 나오는 수자 위치와 그 숫자는 무엇인가요?
np.random.seed(2024)
a=np.random.randint(1,26346,1000)
a
type(a)
a[49]
b=np.where(a>=10000)
b
type(b)
my_index=b[0][49]
a[my_index]
#500보다 작은 숫자들중 가장 마지막으로 나오는 숫자 위치와 그 숫자는 무엇인가?
x=np.where(a<500)
inde=x[0][-1]
a[inde]


a= np.array([20,np.nan,13,24,309])
a+3
type(a)
np.meam(a)
np.nanmean(a) #nan무시 옶션
np.nan_to_num(a,nan=0)


