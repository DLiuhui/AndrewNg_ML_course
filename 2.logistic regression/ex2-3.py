# logistic regression
# use optimization library
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

# activation function
# each x as a vector
def sigmoid(w,x):
    return 1/(1+np.e**(-np.dot(w, x)))

def sigmoid_1p(x):
    return 1/(1+np.e**(-x))

def hypothesis(w,x):
    return sigmoid(w, np.transpose(x))

def predict(w,x):
    return sigmoid(w,np.transpose(x))

if __name__=='__main__':
    # read data
    data = np.loadtxt('ex2data2.txt', delimiter=',')
    n_features = data.shape[1]-1
    n_sets = len(data)
    Y = data[:, n_features]
    X = data[:, 0:n_features]
    x_range = np.zeros(n_features)
    x_min = np.zeros(n_features)
    #归一化
    for i in range(n_features): 
        x_range[i] = np.max(X[:,i]) - np.min(X[:,i])
        x_min[i] = np.min(X[:,i])
        X[:,i] = (X[:,i] - x_min[i])/x_range[i]
    data_pos = data[Y==1]
    data_neg = data[Y==0]
    #plot_data
    plt.figure(1)
    plt.scatter(data_pos[:,0], data_pos[:,1],c='y')
    plt.scatter(data_neg[:,0], data_neg[:,1],c='b')
    plt.show()
    #feature mapping
    x = np.zeros((n_sets, 28))
    for i in range(n_sets):
        idx = 0
        for j in range(7):
            for k in range(j+1):
                x[i][idx] = (X[i][0]**(j-k)) * X[i][1]**(k)
                idx += 1
    #classification, use optimization library
    costF = lambda w: -(np.dot(Y, np.transpose(np.log(hypothesis(w,x)))) + np.dot(1-Y, np.transpose(np.log(1-hypothesis(w,x)))))/len(x)
    w = op.fmin_powell(costF, np.zeros(28), maxiter=5000)
    #draw decision boundrary
    plt.figure(3)
    plt.scatter(data_pos[:,0], data_pos[:,1],c='y')
    plt.scatter(data_neg[:,0], data_neg[:,1],c='b')
    x1_ = np.linspace(0,1,100)
    x2_ = np.linspace(0,1,100)
    x1_, x2_ = np.meshgrid(x1_, x2_)
    y_predict = np.zeros(x1_.shape)
    for i in range(x1_.shape[0]):
        for j in range(x1_.shape[1]):
            x_test = np.array([1,
                               x1_[i,j], x2_[i,j],
                               x1_[i,j]**2, x1_[i,j]*x2_[i,j], x2_[i,j]**2,
                               x1_[i,j]**3, x1_[i,j]**2*x2_[i,j], x1_[i,j]*x2_[i,j]**2, x2_[i,j]**3,
                               x1_[i,j]**4, x1_[i,j]**3*x2_[i,j], x1_[i,j]**2*x2_[i,j]**2, x1_[i,j]*x2_[i,j]**3, x2_[i,j]**4,
                               x1_[i,j]**5, x1_[i,j]**4*x2_[i,j], x1_[i,j]**3*x2_[i,j]**2, x1_[i,j]**2*x2_[i,j]**3, x1_[i,j]*x2_[i,j]**4, x2_[i,j]**5,
                               x1_[i,j]**6, x1_[i,j]**5*x2_[i,j], x1_[i,j]**4*x2_[i,j]**2, x1_[i,j]**3*x2_[i,j]**3, x1_[i,j]**2*x2_[i,j]**4, x1_[i,j]*x2_[i,j]**5, x2_[i,j]**6])
            y_predict[i,j] = sigmoid(w, x_test)
    plt.contour(x1_,x2_,y_predict,1,linewidth=0.5)
    plt.show()