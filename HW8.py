# ADP 표본점수
# 2022년에 실시 된 ADP 실기 시험의 통계파트 표준점수는 평균이 30, 표준편차가 5인 정규분포를
# 따른다고 한다.
# 1) ADP 실기 시험의 통계파트 표준점수의 밀도함수를 그려보세요.
# 2) ADP 수험생을 임의로 1명을 선택하여 통계 점수를 조회했을때 45점 보다 높은 점수를 받았을
# 확률을 구하세요.
# 3) 슬통이는 상위 10%에 해당하는 점수를 얻었다고 한다면, 슬통이의 점수는 얼마인지 계산해보
# 세요.
# 4) 슬기로운 통계생활의 해당 회차 수강생은 16명이었다고 한다. 16명의 통계 파트 점수를 평균
# 내었을 때, 이 평균값이 따르는 분포의 확률밀도 함수를 1번의 그래프와 겹쳐 그려보세요.
# 5) 슬기로운 통계생활 ADP 반 수강생들의 통계점수를 평균내었다고 할 때, 이 값이 38점보다 높게
# 나올 확률을 구하세요.

#1)
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
x = np.arange(15,45,0.1)
y = norm.pdf(x, loc=30, scale=5) # 정규분포에서 x에 해당되는 y값만들기
# loc= 평균 scale= 표준편차
plt.plot(x, y)

#t.pdf 지정된 수에 대응하는 y값
#t.ppf 하위 %값을 반환해줌 ex) norm.ppf(0.95,30,4) # 상위 5% 즉 하위 95%값을 반환해줌
#t.cdf 왼쪽에서부터 지정한 값이 다 나온다
#t.rvs 랜덤값 만들기

#2)
1 - norm.cdf(45, loc=30,scale= 5) # cdf가 45이하가 나올확률을 알려주므로 1-cdf를 하면
# 45점보다 높을 확률이 나온다

#3)
norm.ppf(0.9,30,5) #알고 싶은것은 상위 10% 따라서 입력해주는 값에 하위 90%를 표시하기위해서 0.9

#4) 
x = np.arange(15,45,0.1)
y = norm.pdf(x, loc=30, scale=5)
plt.plot(x, y)

y2 = norm.pdf(x,loc=30,scale=5/np.sqrt(16)) # 표준편차/루트(n)
plt.plot(x,y2,color='red')

#5)
1 - norm.cdf(38, 30, 5/np.sqrt(16))

# Covid 19 발병률
# Covid‑19의 발병률은 1%라고 한다. 다음은 이번 코로나 사태로 인하여 코로나 의심 환자들 1,085
# 명을 대상으로 슬통 회사의 “다잡아” 키트를 사용하여 양성 반응을 체크한 결과이다.
# 키트 \ 실제 양성 음성
# 양성 370 10
# 음성 15 690
# 1) 다잡아 키트가 코로나 바이러스에 걸린 사람을 양성으로 잡아낼 확률을 계산하세요.
# 2) 슬통 회사에서 다잡아 키트를 사용해 양성으로 나온 사람이 실제로는 코로나 바이러스에 걸려
# 있을 확률을 97%라며, 키트의 우수성을 주장했다. 이 주장이 옳지 않은 이유를 서술하세요.
# 3) Covid‑19 발병률을 사용하여, 키트의 결과값이 양성으로 나온 사람이 실제로 코로나 바이러스에
# 걸려있을 확률을 구하세요.
# 카이제곱분포와 표본분산
# 자유도가 𝑘인 카이제곱분포를 따르는 확률변수 𝑋 를
# 𝑋 ∼ 𝜒2(𝑘)
# 과 같이 나타내고, 이 확률변수의 확률밀도함수는 다음과 같습니다.

#1)
370 / 385
#2)
#성능에는 위음성과 위양성도 언급을 해야한다
# 교재정답: 표본으로 뽑힌 집단의 유병률과 모집단의 유병률의 차이가 크다. ?

#3)

sol = (0.01 * (370 / 385)) / (0.01 * (370 / 385) + 0.99 * (10 / 700)) #교재 208P
round(sol,3)

# 다음의 물음에 답하세요.
# 1) 자유도가 4인 카이제곱분포의 확률밀도함수를 그려보세요.
# 2) 다음의 확률을 구해보세요.
# 𝑃 (3 ≤ 𝑋 ≤ 5)
# 3) 자유도가 4인 카이제곱분포에서 크기가 1000인 표본을 뽑은 후, 히스토그램을 그려보세요.
# 4) 자유도가 4인 카이제곱분포를 따르는 확률변수에서 나올 수 있는 값 중 상위 5%에 해당하는
# 값은 얼마인지 계산해보세요.
# 5) 3번에서 뽑힌 표본값들 중 상위 5%에 위치한 표본의 값은 얼마인가요?
# 6) 평균이 3, 표준편차가 2인 정규분포를 따르는 확률변수에서 크기가 20인 표본, 𝑥1, ..., 𝑥20,을
# 뽑은 후 표본분산을 계산한 것을 𝑠21
# 이라 생각해보죠. 다음을 수행해보세요!
# • 같은 방법으로 500개의 𝑠2 들, 𝑠21
# , 𝑠22
# , ..., 𝑠2
# 500 발생시킵니다.
# • 발생한 500개의 𝑠2 들 각각에 4.75를 곱하고, 그것들의 히스토그램을 그려보세요. (히스토그램
# 을 그릴 때 probability = TRUE 옵션을 사용해서 그릴 것)
# • 위에서 그린 히스토그램에 자유도가 19인 카이제곱분포 확률밀도함수를 겹쳐그려보세요.

#1)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
x = np.linspace(0, 20, 1000)
pdf = chi2.pdf(x, 4) # x뒤에는 자유도 # x에 대한 y값 추정
plt.plot(x, pdf) 
plt.xlabel("x")
plt.ylabel("f_X(x; k)")
#2)
chi2.cdf(5, 4) - chi2.cdf(3, 4) ## cdf가 x이하가 나올확률을 알려주므로 cdf(5)-cdf(3)하면 나온다
#3)
np.random.seed(2023)
sample_size = 1000
sample_data = chi2.rvs(4, size=sample_size)
plt.hist(sample_data, bins=50, density=True,
color="lightblue", edgecolor="black");
plt.xlabel("sample")

#4)
chi2.ppf(0.95, 4) # 상위 5% 즉 하위 95%값을 반환해줌

#5)
np.percentile(sample_data, 95) #95번째 백분위수(즉, 상위 5% 경계에 해당하는 값)를 계산하여 반환

#6)
np.random.seed(2023)
n = 20
num_samples = 500
var_samples = []

for i in range(num_samples):
    x = norm.rvs(3, 2, size=n)
    var_samples.append(np.var(x, ddof=1))
scaled_var_samples = np.array(var_samples) * 4.75
plt.hist(scaled_var_samples,
    bins=50, density=True, color="lightblue",
    edgecolor="black"); # hist 설정
plt.xlabel("4.75 * s^2")
plt.ylabel("density")
x = np.linspace(0, max(scaled_var_samples), 1000)
pdf_chi19 = chi2.pdf(x, df=19)
plt.plot(x, pdf_chi19, 'r--', linewidth=2) #빨간선 설정
plt.legend(["histogram", "df 19 chisquare dist"], loc="upper right")
plt.show()


