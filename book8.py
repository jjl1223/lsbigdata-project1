import pandas in pd

mpg=pd.read_csv("data/mpg.csv")
mpg.shape
mpg.head()

import seaborn as sns
import matplotlib.pyplot as plt

#plt.figure(figuresize=(5,4)) 사이즈 조정
sns.scatterplot(data=mpg,x="displ",y="hwy").set(xlim=[3,6],ylim=[10,30])
plt.show()


sns.scatterplot(data = mpg,x='displ',y='hwy',hue='drv')
plt.show()
plt.clf()

# 204
mpg=pd.read_csv("data/mpg.csv")
sns.scatterplot(data=mpg,x='cty',y='hwy')
plt.show()
plt.clf()

midw=pd.read_csv("data/midwest.csv")
midw.head()
mpg['suv'].vaule_counts().index

#막대그래프
mpg.groupby("drv") \
            .agg(mean_hwy=("hwy","mean"))

df_mpg= mpg.groupby("drv",as_index=False) \
            .agg(mean_hwy=("hwy","mean"))
            
df_mpg
sns.barplot(data=df_mpg,x="drv",y="mean_hwy",hue="drv")
            
            
sns.barplot(data=df_mpg.sort_values("mean_hwy"), x="drv",y="mean_hwy",hue="drv" )           
            
df_mpg = mpg.groupby("drv",as_index=False).agg(n=("drv","count"))     

sns.barplot(data=df_mpg,x='drv',y='n')

sns.countplot(data=mpg,x='drv')



            
