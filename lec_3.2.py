import seaborn as sns

var=["a","a","b","c"]
var

seaborn.countplot(x=var)
a=1,2
a
import matplotlib.pyplot as plt # 도표로 보여주게 하기위해서 추가한것 클린하는법은 plt.clf()
plt.show() # 도표로 보여주게 하기위해서 추가한것 
plt.clf()
df=sns.load_dataset("titanic") #이런거를 데이터 프레임이라고한다
df

sns.countplot(data=df, x= 'sex')
plt.clf()

sns.countplot(data=df, x= 'class')
plt.show()
plt.clf()

sns.countplot(data=df, x= 'class',hue ='alive',palette='Set1')
plt.show()
plt.clf()

sns.countplot(data=df, y= 'class',hue ='alive',orient="v")
plt.show()
plt.clf()
import sklearn.metrics
sklearn. metrics.accuracy_score()

from sklearn import metrics
metrics.accuracy_score()


import os
