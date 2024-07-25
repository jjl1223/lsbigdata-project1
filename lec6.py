import numpy as np  
#두개의 벡터 합쳐서 행렬 생성 세로로 쌓기
matrix = np.column_stack((np.arange(1, 5),
np.arange(12, 16)))
#가로로 쌓기
matrix = np.vstack(
    (np.arange(1,5),
    np.arange(12,16))
)

print("행렬:\n", matrix)

np.zeros(5)

np.array([3,4])
np.arange(3,4)



y = np.zeros((5, 4))

np.arange(1,5).reshape(2,2)
# -1 통해서 크기를 자동으로 결정할 수 있음
np.arange(1,15).reshape(2,-1)
# 0.0 에서 99까지 수중 랜덤하게 50개 숫자를 뽑아서 5 by 10 행렬 만드세요.

np.random.seed(2024)

np.random.radint
np.random.choice((1,10),10)
np.random.randint(1,100,50).reshape((5,10)) #reshpae뒤에 그냥 괄호한개만 넣어도 동작하지만 정석은 값이 여러개만 튜플로 넣어주기


np.arange(1,21).reshape((4,5))
mat_a = np.arange(1, 21).reshape((4, 5), order='F')

#인덱싱(위치에 있는값 꺼내오기)
mat_a
mat_a[0,0]
mat_a[1,1]

mat_a[2,3]
mat_a[1:4,1:3]


# 행자리 ,열자 비어있는 경우 전체 행 또는 열 선택
mat_a[3,]
mat_a[3,:] #비어있는거 표시해주기 위해서 :씀 근데 행은 비워놓으면 안돌아간다

mat_a[3,::2] #3행 한칸씩 뛰어넘어서 



# 짝수행만 선택한다

mat_b=np.arange(1,101).reshape((20,-1))
mat_b[1::2,:]

mat_b[[1,4,6,12],]

x= np.arange(1,11).reshape((5,2))*2

x[[True,True,False,False,True],0] #True 값만 뽑아온다 그래서 0열의 True행에 있는 애들만 뽑아옴

mat_b[:,1]  #(1차원) 벡터
mat_b[:,(1,)] #(2차원) 행렬
mat_b[:,[1]] #행렬
mat_b[:,1:2]

#필터링
mat_b[mat_b[:,1] %7 ==0,:]


import numpy as np
import matplotlib.pyplot as plt
# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)  # 0~1사이 랜덤값 발생
print("이미지 행렬 img1:\n", img1)
# 행렬을 이미지로 표시

plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()
# 5행 2열의 행렬 생성
x = np.arange(1, 11).reshape((5, 2)) * 2
print("원래 행렬 x:\n", x)
# 행렬을 전치
transposed_x = x.transpose()
print("전치된 행렬 x:\n", transposed_x)

import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")




import imageio
import numpy as np
# 이미지 읽기
jelly = imageio.imread("jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])

jelly[:,:,0] #R
jelly[:,:,1] #G
jelly[:,:,2] #B
jelly[:,:,3] #투명도



plt.imshow(jelly)
plt.imshow(jelly[:,:,0])
plt.imshow(jelly[:,:,1])
plt.imshow(jelly[:,:,2])
plt.imshow(jelly[:,:,0].transpose()) #행과 열 위치 바꾸므로 옆으로 돌려져서 나온다
plt.axis('off') #축정보 없애기
plt.show()


mat1 = np.arange(1,7).reshape(2,3)
mat2 = np.arange(7,13).reshape(2,3)
mat1.shape
my_array =np.array([mat1,mat2]) #3차원 배열로 합치기
my_array.shape


jelly.shape#88,50,4으로 보이는 이유는 정면으로 보는게 아니라 옆으로 봐서 이렇게 보인다
my_array =np.array([my_array,my_array])



first_slice = my_array[0, :, :]
print("첫 번째 2차원 배열:\n", first_slice)
# 두 번째 차원의 세 번째 요소를 제외한 배열 선택
filtered_array = my_array[:, :, :-1]
print("세 번째 요소를 제외한 배열:\n", filtered_array)

my_array[:,:,[0,2]]
my_array[:,0,:]


mat_x=np.arange(1,101).reshape((5,5,4))
mat_x
mat_x.shape

mat_x[0,:,:]

a=np.array([[1,2,3],[4,5,6]])

a.sum()
a.sum(axis=0) #열끼리 더하기
a.sum(axis=1) #행끼리 더하기



a.mean()
a.mean(axis=0)
a.mean(axis=1)
    

mat_b=np.random.randint(0,100,50).reshape((5,-1))
mat_b

# 가장 큰 수는?
mat_b.max()
mat_b.max(axis=1)

# 행별 가장 큰수는?
mat_b.max(axis=1)

# 열별 가장 큰수는?
mat_b.max(axis=0)


a=np.array([1,3,2,5]).reshape((2,2))
a.cumsum()

a.cumsum(axis=0)

mat_b.cumsum(axis=1)
mat_b.cumprod(axis=1)

mat_b.reshape((2,5,5)).flatten()
mat_b.flatten()

d=np.array([1,2,3,4,5])
d.clip(2,4)

d. tolist()

