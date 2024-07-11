#데이터 타입 
x=15.34
print(x, "는",type(x), "형식입니다.",sep=' ')# sep 자동으로 사이칸 띄어주기


a = [10,20,30,40,50,55,45,34,33] 
b = (10,20,30,40,50,60,70,80,90) #튜플은 다른 명령이나 계산으로 변하지 않는다
#튜플은 b=1,2,3이라고 해도 튜플이 된다
b=1,2,3
b_int =(42)
b_int
b_tp =(42,) # 끝에 ,를 찍어야 int가 아니라 tuple로 인식한다.
b_tp
print(b_int,type(b_int))
print(b_tp,type(b_tp))

b_int+=10
b_tp+=10

a[2]
b[2]

a[2]=25
b[2]=25
b[3:] #지정한 칸(인덱스)부터 오른쪽 끝 까지- 해당칸이상 인덱스 시작은 0
b[:3] #지정한 칸(인덱스)전부터 왼쪽 끝까지- 해당칸 미만
b[0:5] # 지정한 칸 포함하여 오른쪽 숫자만큼간다 
# 해당 인덱스 지정 리스트도 가능

def min_max(numbers):
    return min(numbers) , max(numbers)

result = min_max([1,2,3,4,5])
result = min_max((1,2,3,4,5))
print("Minimum and maximum",result)
#데이터 형식을 뭘로 하든 튜플로 반환

e=3
print(e,type(e))
e+=3

jaejun = {
  
  'name' : 'jaejun',
  "age"  : 25,
  "city" : '제천'
  }
print("jaejun:",jaejun)

jae = {
  
  'name' : 'jaejun',
  "age"  : [20,25],
  "city" : (3,4)
  }
print("jae:",jae)

jae.get("age")[0]
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits) # 중복 'apple'은 제거됨
# 빈 집합 생성
empty_set = set()
print("Empty set:", empty_set)
empty_set.add('f')
print("Empty set:", empty_set)

empty_set.add("apple")
empty_set.add("apple")
empty_set.add("banana")

empty_set.remove("banana")
empty_set.discard("bananan")

other_fruits =fruits.difference(furits)


# 조건문
a=3;
if (a>=2):
  print("a는 2보다 크다")
else:
  print("a는 2 미만이다")
  
  
str_1="가나다"
str_1 = float(str_1)
print("숫자형",str_1,type(str_1))


# 리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)
