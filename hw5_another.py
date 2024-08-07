#hw5번째 다른 값
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform

x = uniform.rvs(loc=3, scale=4, size=20*10000)
x
x.shape
x=x.reshape(-1, 20)
x
x.shape

s_2 = np.var(x, ddof=1, axis=1) # axis 각 행을 기준 (10000,20) 10000행(가로) 20열(세로)
s_2.shape
sns.histplot(s_2, stat='density')
plt.axvline(x = uniform.var(loc=2, scale=4), color = 'green', linestyle = '-', linewidth = 2)
plt.show()
plt.clf()

k_2 = np.var(x, ddof=0, axis=1)
k_2.shape

sns.histplot(k_2, stat='density')
plt.axvline(x = uniform.var(loc=2, scale=4), color = 'green', linestyle = '-', linewidth = 2)
plt.show()
