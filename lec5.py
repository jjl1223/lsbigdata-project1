import numpy as np

np.random.seed(42) # radnom값을 고정시켜주는 역할 안에 숫자를 바꾸면 값이 바뀐다 보통 42가 정석

a= np.random.randint(1,21,10)

print(a)

print(a[1])

a[2:5] #앞에는 자기포함 뒤에는 자기포함 x
a[-2] #맨끝에서 두번째
a[::2]#처음부터 끝까지 2칸식 건너서 뽑아내라

a[0:6:2]

1에서부터 1000사이 3의 배수의 합은?
b=np.arange(0,1001,1)
sum(b[::3])
print(a[[0,2,4]])

np.delete(a,[1,3])
a>9
a[a>9]

np.random.seed(2024) # radnom값을 고정시켜주는 역할 안에 숫자를 바꾸면 값이 바뀐다 보통 42가 정석

a= np.random.randint(1,10000,5)

a[(a>2000) & (a<5000)]

import pydataset

df=pydataset.data('mtcars')
mp_df=np.array(df['mpg'])


#15이상 25이하인 데이터 개수는?
sum((mp_df>=15) & (mp_df<=25))

(mp_df>=15) & (mp_df<=25)

#평균 mpg보다 높은(이상) 자동차 대수
mp_df>=np.mean(mp_df)

sum(mp_df>=np.mean(mp_df))
#15보다 닥거나 22이상인 데이터 개수는?
sum((np_df <15) |(npdf) )

np.random.seed(2024)
a= np.random.randint(1,10000,5)
b= np.array(["a","b","c","d","e"])

a[(a>=2500) & (a<=4000)]
b[(a>=2500) & (a<=4000)]



model_names = np.array(df.index)
#평균 mpg보다 높은(이상) 자동차 이름
model_names[mp_df>=np.mean(mp_df)]




