import numpy as np
# import matplotlib.pyplot as plt
# 利用NN实现简单的分类
# 利用NN分别实现and/or/not
# 最后组合网络，实现xnor
# 简单的神经网络实验，来自视频

# parameters of NN
AND_PARA = np.array([-30,20,20])
OR_PARA = np.array([-10,20,20])
NOT_PARA = np.array([10,-20])
XN_AND_PARA = np.array([10,-20,-20])

# activation function
# each x as a vector
def sigmoid(w,x):
    return 1/(1+np.e**(-np.dot(w, x)))

def sigmoid_1p(x):
    return 1/(1+np.e**(-x))

if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    x = np.c_[np.ones(len(X)), X]
    
    np.set_printoptions(precision=8, suppress=True)
    print('And:{}'.format(sigmoid(AND_PARA, x.T)))
    print('Or:{}'.format(sigmoid(OR_PARA, x.T)))
    print('XN_AND:{}'.format(sigmoid(XN_AND_PARA, x.T)))
    
    # XNOR = Or(AND XN_AND)
    # 隐藏层
    W1 = np.vstack((AND_PARA, XN_AND_PARA))
    a1 = np.dot(W1, x.T)
    z1 = sigmoid_1p(a1)
    # 输出层
    a2 = np.vstack((np.ones(z1.shape[1]), z1))
    W2 = OR_PARA
    a3 = np.dot(W2, a2)
    z3 = sigmoid_1p(a3)
#    W = [0,0]
#    W[0] = np.vstack((AND_PARA, XN_AND_PARA))
#    W[1] = OR_PARA
#    a = x.T
#    z = sigmoid(W[0],a)
#    a = np.vstack((np.ones(z.shape[1]),z))
#    z = sigmoid(W[1],a)
    print('XNOR:{}'.format(z3))