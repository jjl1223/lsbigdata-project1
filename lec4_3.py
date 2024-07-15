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
np.nanmean(a) #nan무시 옵션
np.nan_to_num(a,nan=0)

False
a =None #값이 nan처럼 나오지도 않고 연산이 되지 않는다
b= np.nan
a
b
a+1
b+1

np.isnan(a) #nan값이 있는지 없는지 확인
a = np.array([20, np.nan, 13, 24, 309])
~np.isnan(a)
a_filtered = a[~np.isnan(a)] #a의 true위치값만 출력
a_filtered





str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]


import numpy as np
mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec



combined_vec = np.concatenate((str_vec,mix_vec))  #() 튜플 []list 튜플로 들어오건 리스트로 들어오건 무방하다
#combined_vec = np.concatenate([str_vec,mix_vec])  #() 튜플 []list
combined_vec
type(combined_vec)

col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.row_stack((np.arange(1,5),np.arange(12,16)))# row_stacked는 나중에 np.vstack으로 바뀔 예정
row_stacked

row_stacked = np.vstack((np.arange(1,5),np.arange(12,16)))# row_stacked는 나중에 np.vstack으로 바뀔 예정
row_stacked

vec1=np.arange(1,5)
vec2=np.arange(12,18)

np.resize(vec1, len(vec))

vec1= np.resize(vec1,len((23))
vec1
#전체 5더하기
vec1=np.arange(1,5)+5
#홀수번째
a = np.array([12, 21, 35, 48, 5])
a

a[0::2]
#주어진 벡터에서 최대값을 찾으세요
a.max()

#주어진 벡터에서 중복된 값을 제거한 새로운 벡터를 생성하세요.
np.unique(a)


#주어진 두 벡터의 요소를 번갈아 가면서 합쳐서 새로운 벡터를 생성하세요.
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
a
b

c=np.concatenate((a,b))

x=np.empty(6)

x[[1,3,5]]=a
x
x[1::2]=a
x
x[[0,2,4]]=b
x[0::2]
x














