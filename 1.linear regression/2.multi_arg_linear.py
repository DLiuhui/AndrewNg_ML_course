import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def GradiantDescent(x, y, theta, alpha, iterations=1000):
    loss_history = np.zeros((iterations,1))
    total = len(y)
#    feature_num = x.shape[1]
    for i in range(iterations):
        loss_vec = np.dot(x,theta) - y
        theta_tmp = theta - alpha * np.dot(np.transpose(x), loss_vec) / total
        loss_vec = loss_vec**2
        loss = loss_vec.sum()/2/total
        loss_history[i] = loss
        theta = theta_tmp
    return theta, loss_history

if __name__ == '__main__':
    data_raw = np.loadtxt('ex1data2.txt', delimiter = ',')
    # set X, Y as matrix/vector
    X = data_raw[:,0:data_raw.shape[1]-1]
    Y = data_raw[:,data_raw.shape[1]-1]
    feature_num = X.shape[1]
    total = len(Y)
    
    # feature normalization
    feature_scale = np.zeros((feature_num,1))
    feature_min = np.zeros((feature_num, 1))
    for i in range(feature_num):
        feature_scale[i] = np.max(X[:,i]) - np.min(X[:,i])
        feature_min[i] = np.min(X[:,i])
    for i in range(total):
        for j in range(feature_num):
            X[i,j] = (X[i,j] - feature_min[j])/feature_scale[j]
    
    # plot raw data
    print('multi_arg_linear regression:')
    print('Scatter diagram:')
    fig = plt.figure(1)
    ax = Axes3D(fig)
    ax.scatter(X[:,0], X[:,1], Y)
    plt.show()
    
    print('='*50)
    print('linear regresssion:')
    # X增广
    X_ = np.c_[np.ones(total), X]
    Y = Y[:, np.newaxis]
    theta = np.random.random((feature_num+1,1))
    # learning rate
    alpha = 0.01
    iterations = 3000
    # GD
    theta, loss_history = GradiantDescent(X_, Y, theta, alpha, iterations)
    # plot loss function
    fig = plt.figure(2)
    plt.plot(np.arange(iterations), loss_history)
    # plot convergence
    fig = plt.figure(3)
    ax = Axes3D(fig)
    x1_test = np.linspace(0,1,100)
    x2_test = np.linspace(0,1,100)
    x1_test, x2_test = np.meshgrid(x1_test, x2_test)
    y_test = theta[0] + x1_test*theta[1] + x2_test*theta[2]
    ax.plot_surface(x1_test, x2_test, y_test, linewidth=1, rstride=1, cstride=1, cmap='rainbow')
    




