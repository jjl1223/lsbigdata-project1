import pandas as pd
import numpy as np

test1= pd.DataFrame({"id": [1,2,3,4,5],
                     "midterm":[60,80,70,90,85]})
test2= pd.DataFrame({"id": [1,2,3,40,5],
                     "final":[70,83,65,95,80]})
type(test1)                     
                     
total = pd.merge(test1,test2,how="left",on="id") #왼쪽 기준으로 id기준으로 통합
total
#left join                     
test1_= pd.DataFrame({"id": [1,2,3,4,5],
                     "midterm":[60,80,70,90,85]})
test2_= pd.DataFrame({"id": [1,2,3,40,5],
                     "final":[70,83,65,95,80]})
#right join                     
total = pd.merge(test1_,test2_,how="right",on="id")
total

#inner join #아이디 맞는것만 합치기
total = pd.merge(test1,test2,how="inner",on="id")
total

#outer join #전부합치기
total = pd.merge(test1,test2,how="outer",on="id")
total


name=pd.DataFrame({"nclass":[1,2,3,4,5],
                   "teacher": ["kim","lee","park","choi","jung"]             })
name
exam=pd.read_csv('data/exam.csv')

pd.merge(exam,name,how="left",on="nclass")


#데이터 세로로 쌓는 방법
score1= pd.DataFrame({"id": [1,2,3,4,5],
                     "score":[60,80,70,90,85]})
score2= pd.DataFrame({"id": [6,7,8,9,10],
                     "score":[70,83,65,95,80]})

score1
score2
score_all=pd.concat([score1,score2])
score_all

test1
test2

score_all=pd.concat([score1,score2],axis=1) #contact로 옆으로 쌓기
score_all

fuel = pd.DataFrame({"f1"   : ["c","d","e","p","r"]
                                })



                     
