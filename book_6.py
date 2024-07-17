import numpy as np
import pandas as pd

#데이터 전처리 함수
#query() 행추출
#df[] 열(변수) 추출
#sort_value() #정렬



exam=pd.read_csv('data/exam.csv')
type(exam)
exam.query("nclass ==1")
#exam[exam["nclass"]==1]
exam.query('math>50')
exam.query('math<50')

exam.query("english>=50")
exam.query("english<=80")

exam.query("nclass ==1 & math>=50")
exam.query("nclass ==2 and english >=80")

exam.query('math>=90 or english>=90')
exam.query('math>=90 | english>=90')

exam.query('english <90 or science <50')
exam.query('nclass ==1 or nclass ==3 or nclass ==5')
exam.query('nclass in [1,3,5]')
exam.query('nclass not in [1,3,5]')

type(exam["math"]) #pandas series
exam[["id","math"]]

type(exam[["id","math"]]) #pandas data frame

exam.drop(columns = "math") # 없애기

exam.query('nclass==1')['english']
exam.query('nclass==1')\
[['english',"math"]]\
.head()


#정렬하기
exam.sort_values('math')
exam.sort_values(["nclass",'math'], ascending =[True,False])
#숙제: p144,p153,p158

#변수추가 
exam.assign(total = exam['math'] + exam["english"]+exam["science"])

exam2=exam.assign(total = exam["math"] +exam["english"]+exam["science"],
                    mean = (exam["math"]  +exam["english"]+exam["science"] )/3
                    )\
                    .sort_values("total",ascending=False)

#요약을 하는 .agg()
#그룹을 나눠 요약을 하는 .groupby() + agg() 콤보
exam2
exam2.agg(mean_math=("math","mean"))
exam2.groupby("nclass")\
     .agg(mean_math = ("math","mean"))
                
exam2.groupby('nclass')\
        .agg(mean_math = ('math','mean')
        sum_math = ('math','sum')
        median_math = ('math','median')
        n = ('nclass','count')
        )
        
        
        

import pydataset

df= pydataset.data("mpg")

mpg_data= mpg.query('category' =='suv')




