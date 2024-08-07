# 지도 시각화 book11
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    
import seaborn as sns

import json
geo_seoul=json.load(open('bigfile/SIG_Seoul.geojson',encoding="UTF-8"))
#json파일들은 딕셔너리안에 딕셔너리가 이루어져있다 즉, key와 key내용으로 이루어져있다 
geo_seoul.keys()
type(geo_seoul)
len(geo_seoul)



len(geo_seoul["features"])

len(geo["features"][0])
type(geo_seoul["features"]) 
type(geo_seoul["features"][0]) #딕셔너리 확인

geo_seoul["features"][0].keys()
geo_seoul["crs"].keys()

geo["features"][5]["properties"]
geo["features"][5]["properties"]['SIG_KOR_NM']
geo["features"][0]["geometry"]
# 중괄호 없애서 딕셔너리 리스트에 접근하기
geo["features"][0]["geometry"]["coordinates"]

coordinates_list=geo["features"][5]["geometry"]["coordinates"]


#이렇게 하면 대괄호가 하나씩 없어진다
len(coordinates_list[0])
len(coordinates_list[0][0])

#리스트로 정보 빼오기
coordinate_array=np.array(coordinates_list[0][0])
x=coordinate_array[:,0]
y=coordinate_array[:,1]

plt.plot(x,y,color="black")  
plt.plot(x[::2],y[::2])# 성능을 올라가게 하기 위해서 일부만 그리게 하기
plt.show()
plt.clf()

np.arange(0,25)
---------------------------------------------------------------------------------
#조별 코드
#1번 데이터프레임 만들기 데이터 프레임만 된다
geo_data_main=pd.DataFrame()
geo_data=pd.DataFrame()
for x in np.arange(0,25):
    c_name=geo_seoul["features"][x]["properties"]['SIG_KOR_NM']
    coordinates_list=geo_seoul["features"][x]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinates_list[0][0])
    c_1x=coordinate_array[:,0]
    c_1y=coordinate_array[:,1]
    
    geo_data=pd.DataFrame({
    "name":c_name,
    "x" : c_1x,
    "y" : c_1y
    
    
    })    
    geo_data_main=pd.concat([geo_data_main,geo_data],ignore_index=True)

geo_data


plt.plot(geo_data_main['x'],geo_data_main['y'])
plt.show()
plt.clf()


#2번 그림은 되는데 데이터프레임이 아님    
    
geo_mex=[]
geo_mey=[]
geo_name=[]
for x in np.arange(0,25):
    gu_name=geo_seoul["features"][x]["properties"]['SIG_KOR_NM']
    coordinates_list=geo_seoul["features"][x]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinates_list[0][0])
    geo_mex.append(coordinate_array[:,0])
    geo_mey.append(coordinate_array[:,1])
    geo_name.append(geo["features"][x]["properties"]['SIG_KOR_NM'])
for x in np.arange(0,25):
    plt.plot(geo_mex[x],geo_mey[x])
    plt.show()
    
plt.clf() 




---------------------------------------------------------
# 선생님 코드


#구이름 만들기

gu_name=geo_seoul["features"][0]["properties"]["SIG_KOR_NM"]
gu_name
gu_name=list()
for i in range(25):
    gu_name.append(geo_seoul["features"][i]["properties"]['SIG_KOR_NM'])
    
gu_name    
    
#구 x값 y값 받아서 dataframe으로 만들기

def make_seouldf(num):
    gu_name=geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]

    return pd.DataFrame({"gu_name":gu_name, "x": x, "y": y})

make_seouldf(1)

result=pd.DataFrame({})
for i in range(25):
    result=pd.concat([result, make_seouldf(i)], ignore_index=True)    

result
#서울 지도 그리기
result.plot(kind="scatter",x="x",y="y",style="o",s=1,pallete="")
plt.show()
plt.clf()

# plot은 hue가 되지 않아서 sns로 해주기
sns.scatterplot(data=result,x="x",y="y",hue="gu_name",s=2)

#강남구 색 다르게 그리기
#강남 구별 새로지정해주고
gangnam_df=result.assign(is_gangnam=np.where(result["gu_name"]=="강남구","강남","안강남"))

#hue는 그목차별로 색다르게 legend는 옆에 칸 안뜨게 하기
sns.scatterplot(data=gangnam_df,x="x",y="y",legend=False,hue="is_gangnam",s=2)

#palette로 그색 지정해주기
sns.scatterplot(data=gangnam_df,x="x",y="y",legend=False,palette={ "안강남": "grey" ,"강남": "red" },hue="is_gangnam",s=2)
gangnam_df["is_gangnam"].unique()

plt.show()
plt.clf()

-------------------------------------------------------------------
#책 다시 시작

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    
import seaborn as sns
import json

geo_seoul=json.load(open("./bigfile/SIG_Seoul.geojson",encoding="UTF-8"))
geo_seoul["features"][0]["properties"]

df_pop=pd.read_csv("bigfile/Population_SIG.csv")
df_pop.head()

df_seoulpop=df_pop.iloc[1:26]
df_seoulpop["code"]=df_seoulpop["code"].astype(str)
df_seoulpop.info()

#!pip install folium
#지도는 구글지도나 OpenStreetMap 활용 folium는 OpenStreetMap 씀
import folium
center_x=result["x"].mean()
center_y=result["y"].mean()
my_map=folium.Map(location=[37.55,126.97],zoom_start=7,tiles="cartodbpositron")

my_map.save("map_seoul.html")

#코로플릿
folium.Choropleth(
    geo_data=geo_seoul,
    data=df_seoulpop,
    columns=("code","pop")
    
    
)

