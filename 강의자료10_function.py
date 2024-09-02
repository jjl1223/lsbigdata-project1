import numpy as np
def g(x=3):
    result = x+1
    return result

g()

import inspect
print(inspect.getsource(g))
def g(x=3):
    result = x + 1
    return result
np.array([1,2,3])

# if .. else 정식
x=3
if x > 4:
    y=1
else:
    y=2
print(y)       

# if else 축약
y=1 if x > 4 else 2
y

#리스트 컴프리헨션
x=[1,-2,3,-4,5]
result = ["양수" if value >0 else "음수" for value in x]

result

# 조건 3가지 넘파이 버전
import numpy as np
x = np.array([1, -2, 3, -4, 0])
conditions = [
x > 0,
x == 0,
x < 0
]
choices = [
"양수",
"0",
"음수"
]
result = np.select(conditions, choices,x)
print(result)

#for loop
for i in range(1, 4):
    print(f"Here is {i}")
#for loop 리스트 컴프리헨션
[f"Here is {i}" for i in range(1,4)]

name="남규"
age="31 (진)"
greeting = f"Hello, my name is {name} and I am {age} years old."
print(greeting)


names = ["John", "Alice"]
ages = np.array([25, 30]) # 나이 배열의 길이를 names 리스트와 맞춤

# zip() 함수로 names와 ages를 병렬적으로 묶음
zipped = zip(names, ages)
# 각 튜플을 출력
for name, age in zipped:
    print(f"Name: {name}, Age: {age}")

# while 문
i = 0
while i <= 10:
    i += 3
    print(i)

i = 0
while True:
    i += 3
    if i > 10:
        break
    print(i)


import pandas as pd

data = pd.DataFrame({'A' : [1,2,3],
        'B' : [4,5,6]

})
data

data.apply(max,axis=0)
data.apply(max,axis=1)

# const값이 입력되면 입력된값으로 입력이 안되면 디폴트값인 3으로 적용된다
def my_func(x,const=3):
    return max(x) **2 + const

my_func([3,4,10],5)

data.apply(my_func,axis=1)

data.apply(my_func,axis=0,const=5)

array_2d = np.arange(1,13).reshape((3,4),order="F")
print(array_2d)

np.apply_along_axis(max,axis=0,arr=array_2d)

# 함수 환경
# 저장공간이 def로 만들어진 저장공간과
# 파이썬 전체 저장공간과 다르다
# 그래서 def에 1을 지정해줘도 def 밖 외부 y값은 변하지 않는다
y=2
def my_func(x):
    y=1
    result= x+y
    return result

my_func(3)
print(y)

#이렇게 global을 지정하면 def안에서의 y가아닌
# 외부환경에서의 y를 사용한다는 뜻이다
y=2
def my_func(x):
    global y # 외부환경 y를 사용하겠다

    def my_f(k):
        return k**2


    y= my_f(x) +1
    result= x+y
    return result

# 안돌아간다 왜? def my_func(x)안에서만 지정된 함수이므로 global 즉 외부환경에서는 바로 접근이 불가능하다
my_f(3) 


print(y)
my_func(3)
print(y)


# 입력값이 몇 개일지 모를때 *를 붙임
# 함수는 return이 나오면 무조건 끝난다
# 따라서 for구문안에 return이 들어가면 한번 돌아가고 함수가 바로 끝난다
def add_many(*args):
    result =0
    for i in args:
        result = result +i
    return result
    
add_many(1,2,3)   

# 

def first_many(*args):
    return args
    
first_many(1,2,3)
# args를 list로 받아온다

def add_mul(choice, *my_input):
    if choice == "add":
        result = 0
        for i in my_input:
            result = result +i
    elif choice == "mul":
        result =1
        for i in my_input:
            result = result * i
    return result

add_mul("add",5,4,3,1)

# 별표 두개 (**)는 입력값을 딕셔너리로 만들어줌
def my_twostars(choice, **kwargs):
    if choice == "fist":
        result = print(kwargs["age"])
    elif choice == "second":
        result = print(kwargs["name"])
    else:
       return print(kwargs)
    

    