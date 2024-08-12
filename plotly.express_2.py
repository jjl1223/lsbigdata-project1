import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color = "species",
    size_max=10, #기본점 크기 설정
    #trendline="ols"# p.134 회귀직선 그려주기
)

fig.show()

fig.update_layout(


    title={"text":"<span style ='color:red;font-weight:bold;'>팔머펭귄</span>",
    'x' :0.5,
    'y' :0.5
        }
)

fig



