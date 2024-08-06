import pandas as pd
import numpy as np
gsheet_url = "https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/gviz/tq?tqx=out:csv&sheet=sheet2"
df = pd.read_csv(gsheet_url)
df.head()
np.random.seed(2019)
np.random.choice(np.arange(1,21),10,False)

a=np.random.choice(np.arange(1,29),2,False)

b=np.random.choice(df["이름"],2,False)
b
df["이름"][a[0]]
df["이름"][a[1]]


matrix = np.full((5, 3), 7)
