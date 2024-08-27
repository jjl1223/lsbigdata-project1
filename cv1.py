import numpy as np
import pandas as pd

df=pd.read_clipboard() # 클립보드 읽어오기
df
df=df.sort_values("성별",ignore_index=True)
df_f=df.query("성별=='F'")
df_m=df.query("성별=='M'")

np.random.seed(20240827)
team1_F=np.random.choice(df_f["이름"],6,replace=False)
team1_M=np.random.choice(df_m["이름"],6,replace=False)
np.random.choice(12,6,replace=False)


team1_f=np.random.choice(13,6,replace=False)
team1_m=np.random.choice(11,6,replace=False)

team1=pd.concat([df_f.iloc[team1_f,],df_m.iloc[team1_m,]])
team1
team2=df.drop(team1.index)
team2
