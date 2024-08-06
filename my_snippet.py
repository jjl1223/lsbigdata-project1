#pandas
snippet pd
	import pandas as pd
#numpy
snippet np
	import numpy as np
#seaborn
snippet sns
	import seaborn as sns
#plt
snippet plt
	import matplotlib.pyplot as plt
#nacheck
snippet nacheck
	${1:my_df}["${2:변수명}"].isna().sum()
#그림보기
snippet pshow
	plt.show()
	plt.clf()
