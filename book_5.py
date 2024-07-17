import pandas as pd
import numpy as np

exam= pd.read_csv("data/exam.csv")
exam.head()
exam.head(10)
exam.tail(10)

exam.shape
exam.info()

exam.describe()

#메서드 vs. 속성(어트리뷰트)

type(exam) #데이터 프레임이 여러개있는데 이거는 판다스 데이터 프레임이다

var=[1,2,3] #list

exam2=exam.copy()
exam2

exam2.rename(columns={"nclass": 'class'}) #이대로는 파일자체에서 바뀌지 않음
exam2= exam2.rename(columns={"nclass": 'class'}) #이렇게 해야 바뀜

exam2["total"] = exam2["math"] + exam2["english"] +exam2["science"]
exam2.head()
#200이상 pass 미만 fail
exam2["test"]= np.where(exam2["total"]>=200,"pass","fail")
exam2.tail(20)

exam2["test"].value_counts()
import matplotlib.pyplot as plt
exam2["test"].value_counts().plot.bar()
plt.show()
plt.clf()
count_test=exam2["test"].value_counts().plot.bar(rot=0)
plt.show()

df= pd.DataFrame
#200이상 A 100이상 B 100미만 C
exam2["test2"]=np.where(exam2["total"]>=200,"A",
               np.where(exam2["total"]>=100,"B","C")                )

exam2.head()

exam2["test2"].isin(["A","C"])

np.random.randint(1,21,10) #중복포함
np.random.choice(np.arange(1,21),10,False) #중복이 나오지 않는다
np.random.choice(np.arange(1,4),10,True,np.array([2/5,2/5,1/5])) #확률을 다르게











