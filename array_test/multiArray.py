import numpy as np

#-----------activate function------------#
def step_func(x):
    return np.array(x>0,dtype=np.int)

def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

#----------------1층----------#

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X,W1)+1
Z1 = sigmoid(A1)

print('-----------1 floor------------')
print(A1)
print(Z1)
print('\n')

#----------------2층----------#
W2 = np.array([[0.1,0.4],[0.2, 0.5],[0.3, 0.6]])
B2 = np.array([0.1,0.2])

A2 = np.dot(Z1,W2)+B2
Z2 = sigmoid(A2)

print('--------2 floor--------------')
print(A2)
print(Z2)
print('\n')


#-----------------3층---------------#

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2,W3)+B3
Y = identity_function(A3)

print('----------3 floor-------------')
print(A3)
print(Y)
