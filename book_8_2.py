import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
economics = pd.read_csv('data/economics.csv')
economics.tail()

economics.info()

sns.lineplot(data=economics,x="date",y="unemploy")

plt.show()
plt.clf()


economics["date2"] =pd.to_datetime(economics["date"]) #object 형태를 날짜시간타입으로 바꾼다

economics.info()

economics[["date","date2"]]
economics["date2"].dt.year
economics["date2"].dt.day
economics["date2"].dt.month_name()
economics["date2"].dt.quarter # 날짜시간타입에서 월을 가지고 분기 구별
economics["quarter"]=economics["date2"].dt.quarter

#각 날짜는 무슨 요일인가?
economics["date2"].dt.day_name()

economics[["date2","quarter"]]

economics["date2"] + pd.DateOffset(days=30) #날짜더하기
economics["date2"] + pd.DateOffset(months=1)#달 더하기
economics["date2"].dt.is_leap_year #윤년체크

economics["year"] = economics["date2"].dt.year

sns.lineplot(data=economics,x="year",y="unemploy")

plt.show()
plt.clf()

sns.lineplot(data=economics,x="year",y="unemploy",errorbar = None) 
sns.scatterplot(data=economics,x="year",y="unemploy",errorbar = None) 
plt.show()
plt.clf()

a=economics.groupby('year')\
           .agg( mean=("unemploy","mean"),
                 std=("unemploy","std") , # 표준편차 구하기
                 n= ("unemploy",'count') # 개수 세기
                 )
               
a.head()    
a=a.reset_index()
mean +1.96*std/sqrt(n)   

a["left_ci"] = a['mean'] -1.96*a["std"] / np.sqrt(a["n"])  # 신뢰수준 범위 왼쪽 끝값
a["right_ci"] =  a['mean'] +1.96*a["std"] / np.sqrt(a["n"]) # 신뢰수준 범위 오른쪽 끝값

plt.plot(a["year"],a["mean"],color="black")
plt.scatter(a["year"],a["left_ci"],color="blue",s=2)
    plt.show()
plt.clf()             
                 
