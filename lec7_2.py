fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]
print("과일 리스트:", fruits)
print("숫자 리스트:", numbers)
print("혼합 리스트:", mixed)

#빈리스트 생성
empty_list =[]
empty_list =list()
#초기값 을 가진 리스트 생성
number = [1,2,3,4,5]
range_list = list(range(5))
range_list

range_list[3] = "LS 빅데이터 스쿨"
range_list
#두번째 원소 변경
range_list[1] = ["1st","2nd","3RD"]

range_list[1][2]

#리스트 내포(comprehension)
#1. 대괄호로 쌓여져 있다 리스트다
#2. 넣고 싶은 수식표현을 x를 사용해서 표현
#3. for..in을 사용해서 원소정보 제공
list(range(10))

squares = [x**2 for x in range(10)]

my_squares=[x**3 for x in [3,5,2,15]] # in안에 있는 값을 x에 넘겨서 리스트가 끝날때까지 반복 그리고 그 x를 다시 앞에 연산

import numpy as np
my_squares=[x**3 for x in np.array([3,5,3])]


import pandas as pd
exam = pd.read_csv("data/exam.csv")
exam["math"]

my_squares=[x**3 for x in exam["math"]]

#리스트 합치기
3+2
"안녕" +"하세요"
"안녕" *3
#리스트 연결
list1 = [1,2,3]
list2=[4,5,6]

combind_list= list1+list2

(list1 *3) + (list2 *5)



numbers = [5,2,3]
repeated_list1 = [x for x in numbers for y in range(3)] # y가 범위안에 값을 반복 그리고 그 y값이 넘버로 들어가게된다
repeated_list2 = [x for x in numbers for y in range(3)]
repeated_list = [x for x in numbers for _ in [1,1,1,1]]
repeated_list


# _의 의미
#1. 앞에 나온 값을 가리킴
5+4
_+6 # _는 9를 의미한다


#for 루프 문법
#for in 범위:
#    작동방식

for x in [4,1,2,3]:
    print(x)

for i in range(5):
    print(i**2)

#리시트를 하나 만들어서
# for 루프를 사용해서 2,4,6,8,...,20의 수를
# 채워넣기

emlist=[]
emlist=list()
slist=[]
slist.append(2)
slist.append(4)
slist.append(6)

a= (range(10))

for x in a :
    emlist.append(x*2)
emlist = [0] *10 #리스트 10개 곱해서 만들기
emlist

for x in range(10):
    emlist[x] =x
    
emlist

for x in range(10):
    mylist[x] =x
    
mylist

# mylist_b에 있는  홀수번째 위치 숫자만 가져오기

mylist_b =[2,4,6,80,10,12,24,35,23,20]


mylist = [0]*5


for x in range(5):
    mylist[x] = mylist_b[x*2]

mylist

#리스트 내포(컴프리헨션으로 바꾸는 방법)
# 바깥은 무조건 대괄호로 묶어줌 : 리스트 반환하기 위해서
#for 루프는 :는 생략한다.
#실행 부분을 먼저 써준다.
#결과를 받는 부분 제외시킴

mylist=[]
for i in range(1,11):
    mylist.append(i)
mylist
#위에 꺼와 같은것
[mylist.append(i) for i in range(1,11)]
    
mylist

for i in range(3):
    for j in range(4):
        print(i,j)
a


for i in [0,1,2]:
    for j in [0,1,2,3]:
        print(i,j)
a 




numbers = [5,2,3]
for i in numbers:
    for j in range(4):
        print(i,j)
#위에꺼와 같은것        
repeated_list1 = [x for x in numbers for y in range(3)] 

numbers = [5,2,3]
for i in numbers:
    for j in [3,4,5]:
        print(i)
#위에꺼와 같은것        
repeated_list1 = [x for x in numbers for y in [3,4,5]] 

#원소 체크
fruits = ["apple", "banana","cherry"]
fruits
"banana" in fruits

for x in fruits:
    x == "banana"
mylist=[]
for x in fruits:
    mylist.append(x == "banana")
    
mylist    
type(fruits)
fruits = ["apple","apple","banana","cherry"]
x=np.where(fruits=="banana") # list는 np,where가 안됨

fruits=np.array(fruits)
x=np.where(fruits=="banana") 

np.where(fruits=="banana")[0][0]
int(np.where(fruits=="banana")[0][0])

#원소 거꾸로 서주는 reverse()
fruits = ["apple","apple","banana","cherry"]

fruits.reverse()
fruits

#원소 맨끝에 붙여주기
fruits.append("pineapple")
fruits

fruits.insert(2,"test")

fruits

#원소 제거
fruits = np.array(["apple","apple","banana","cherry"])
items_to_remove =np.array(["banana","apple"])

mask = ~np.isin(fruits, items_to_remove)

mask = ~np.isin(fruits,["banana","apple"])
# 마스크를 사용하여 항목 제거
fruits = np.array(["apple","apple","banana","cherry"])
fiilter=fruits[mask]

fiilter


