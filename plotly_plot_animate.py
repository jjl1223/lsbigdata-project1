import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np

import os

cwd= os.getcwd()# 내 워킹디렉토리 확인
cwd
# 현재 워킹디렉토리 바꾸기
# os.chdir('c:\\Users\\USER\\Documents\\LS빅데이터스쿨\\lsbigdata-project1')
# 주석처리 : ctrl +/
# plotly 이용해서 점과 선그리기
df_covid19_100=pd.read_csv("data/df_covid19_100.csv")
df_covid19_100.info()

fig = go.Figure()
fig.show()
margins_P = {'t' : 50,
             'b' : 25,
             'l' : 25,
             'r' : 25}
#데이터는 딕셔너리로 만들고 각각 다른 것들은 튜플로 저장한다
fig = go.Figure(
    data = [{
        "type" : 'scatter',
        "mode" : "markers",
        "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR","date"],
        "y": df_covid19_100.loc[df_covid19_100["iso_code"] =="KOR","new_cases"],
        "marker" : {"color":"red"}
    },
    
       {
            "type" : 'scatter',
            "mode":"lines",
            'x':df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            'y':df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
           "line" : {"color":"blue",'dash':'dash'}

        }
        ],
    layout = {
    'title' : "코로나 19 발생현황 ",
    "xaxis" : {"title":"날짜",'showgrid' : False},
    "yaxis" : {"title":"확진자수"},
    "margin" : margins_P
    
    
    
    
}
    
    
    
)

fig.show()



             
