import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

#-------------------------넘파이 기본연산----------------------#
#############4번째 수정 ##############

#### 5번째 수정 #######
A = np.array([[1,2,5],[3,4,6],[7,7,1],[9,1,5]])
B = np.array([4,6])
print(A.shape)
print(A.dtype)

print(A[0])
print(A[1][1])
print(A[3][1])

for row in A:
    print(row)

A= A.flatten()
print(A)

print(A[np.array([0,1,2])])
X=np.array([[51,55],[14,19],[0,4]])
print(X)
print(X[np.array([2])])

print(X>15)
print(X[X>15])

#---------------------------넘파이 기본연산 ---------------#

#data set
x =np.arange(0,20,0.1)
y1= np.sin(x)
y2 = np.cos(x)

#그래프 그리기
plt.plot(x, y1, label='sin')
plt.plot(x, y2, label='cos', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin&cos')
plt.legend()

#-------------------------이미지 불러오기--------------------#
img = imread('sojin.jpg')

#### 코드 수정함 #########
### dkdkdkdkdkdkflsadfaskldfjalksdfjkl####

plt.imshow(img)
plt.show()
