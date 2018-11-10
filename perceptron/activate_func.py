import numpy as np
import matplotlib.pylab as plt

def step_func(x):
    return np.array(x>0,dtype=np.int)

def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def relu(x):
    return np.maximum(0, x)

#-----------step function----------#
x=np.arange(-5.0,5.0,0.1)
y= step_func(x)
plt.plot(x,y, label='step')


#----------sigmoid function---------#
sx = np.arange(-5.0, 5.0, 0.1)
sy = sigmoid(sx)
plt.plot(sx,sy, label='sigmoid', linestyle='--')


#-----------relu function-----------#
rx = np.arange(-0.5, 5.0, 0.1)
ry = relu(rx)
plt.plot(rx,ry, label='relu', linestyle='-.')



#-----------graph--------------------#
plt.xlabel('x')
plt.ylabel('y')
plt.title('activate function')
plt.legend()
plt.ylim(-0.1, 1.1)
plt.show()
