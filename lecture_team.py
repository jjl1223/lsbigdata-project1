import pandas as pd
import numpy as np    

pjd=pd.read_excel("data/pjd.xlsx")
low_wage=pjd["최저시급"]
cpi_2020=pjd["인플레율"]
pjd.shape
pjd.info()
pjd.head()
pjd.describe()     
conv=cpi_2020[0]/cpi_2020[9]

cpi=cpi_2020/conv

pjd=pjd.assign(인플레율_2011=pjd["인플레율"]/conv) #새로운 변수생성
pjd_1.rename(columns={"인플레율_2011": '인플레율_보정'}) #이름변경
pjd_1=pjd.drop(columns = "인플레율_보정")
pjd.query("최저시급>=5000") #행들 필터링

pjd_1.sort_values(["최저시급",'인플레율'], ascending =[True,True]) #정렬
pjd.groupby("연도")\
     .agg(pjd_최저시급 = ("최저시급","mean")) #그룹변수 기준 요약


