import pandas as pd
import statsmodels.api as sm
import numpy as np
# 1 종속변수: 백혈병 세포 관측 불가 여부 (REMISS), 1이면 관측 안됨을 의미

# 독립변수:

# 골수의 세포성 (CELL)
# 골수편의 백혈구 비율 (SMEAR)
# 골수의 백혈병 세포 침투 비율 (INFIL)
# 골수 백혈병 세포의 라벨링 인덱스 (LI)
# 말초혈액의 백혈병 세포 수 (BLAST)
# 치료 시작 전 최고 체온 (TEMP)

# 문제 1.

# 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요.

# 텍스트 파일 경로
txt_file_path = 'data/leukemia_remission.txt'
csv_file_path = 'data/leukemia_remission.csv'

# 탭으로 구분된 데이터를 읽어서 DataFrame으로 변환
data = pd.read_csv(txt_file_path, delimiter='\t')

# CSV 파일로 저장
data.to_csv(csv_file_path, index=False)

#파일 불러오기
admission_data= pd.read_csv('data/leukemia_remission.csv')


# REMISS의 로그오즈에 대한 모델 만들기 REMISS(독립변수) ~나머지 전부 종속변수
model = sm.formula.logit("	REMISS ~ CELL + SMEAR + INFIL + LI +BLAST +TEMP ", data=admission_data).fit()

# 표 띄우기
print(model.summary())
# 문제 2.

# 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명하시오.


#검정 통계량 구하는법 (LL-Null-Log-Likelihood)
stat_value=-2*(-17.186 - (-10.797))

from scipy.stats import chi2
# 검정 통계량 이용해서 p-value 구하기
1-chi2.cdf(stat_value, df=6) # df=변수갯수
# 유의수준 0.05하에서 LLR p-value가 0.0467로 작으므로 통계적으로 유의하다.

# 문제 3.

# 유의수준이 0.2를 기준으로 유의한 변수는 몇개이며, 어느 변수 인가요?
# LI와TEMP가   P>|z|가 0.2보다 크므로 2가지 변수가 유의하다


# 문제 4. 다음 환자에 대한 오즈는 얼마인가요?

# CELL (골수의 세포성): 65%

# SMEAR (골수편의 백혈구 비율): 45%

# INFIL (골수의 백혈병 세포 침투 비율): 55%

# LI (골수 백혈병 세포의 라벨링 인덱스): 1.2

# BLAST (말초혈액의 백혈병 세포 수): 1.1세포/μL

# TEMP (치료 시작 전 최고 체온): 0.9


# coef값 불러오는 법
coefficients = model.params
#오즈값구하기 0.0381748712694388
my_odds = np.exp(64.2581 +30.8301*0.65 + 24.686316*0.45 -24.9745*0.55 +4.3605*1.2 -0.0115*1.1 -100.1734*0.9)
my_odds
#문제 5. 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마인가요?
my_odds / (my_odds+1) # 백혈병 관측되지 않을 확률: 0.036771137816849764
#3.7%

#문제 6. TEMP 변수의 계수는 얼마이며, 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명하시오.
coefficients[6] 
# 온도가 0.1증가할때마다 오즈값이 .0009118819655545162만큼 곱해진다
# 온도가 0.1증가할때마다 -99.99546000702375%만큼 변한다

np.exp(-10)*100-100

np.exp(86.89-10)

#문제 7. CELL 변수의 99% 오즈비에 대한 신뢰구간을 구하시오.

np.exp(30.8301)
# x_hat+-z0.005*std.err
from scipy.stats import norm
np.exp(30.8301)-norm.ppf(0.995, loc=0, scale=1)*52.135 
np.exp(30.8301)+norm.ppf(0.995, loc=0, scale=1)*52.135 


# 문제 8. 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후,
#  50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.


y=admission_data["REMISS"]
x=admission_data.drop("REMISS", axis=1)
y_hat = model.predict(x)

y_hat / (y_hat+1)
