# logistic regression
import numpy as np
import matplotlib.pyplot as plt

# activation function
# each x as a vector
def sigmoid(w,x):
    return 1/(1+np.e**(-np.dot(w, x)))

def sigmoid_1p(x):
    return 1/(1+np.e**(-x))

def hypothesis(w,x):
    return sigmoid(w, np.transpose(x))

def costFunction(x,y,w):
    return -(np.dot(y, np.transpose(np.log(hypothesis(w,x)))) + np.dot(1-y, np.transpose(np.log(1-hypothesis(w,x)))))/len(x)

def derivateCostFunc(x,y,w):
#    print(np.dot(y - hypothesis(w, x), x))
    return (np.dot(hypothesis(w, x) - y, x))/len(x)

def gradientDescent(x,y,iteration=100, learning_rate = 0.01):
    costFunc = np.zeros(iteration)
#    w = np.zeros(x.shape[1])
    w = np.random.rand(x.shape[1])
    for i in range(iteration):
        costFunc[i] = costFunction(x,y,w)
        tmp_w = derivateCostFunc(x,y,w)
        w = w - learning_rate * tmp_w
    return w,costFunc

def predict(w,x):
    return sigmoid(w,np.transpose(x))

if __name__=='__main__':
    # read data
    data = np.loadtxt('ex2data1.txt', delimiter=',')
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
    #sigmoid(0)
    print(sigmoid_1p(0))
    #classification
    iteration = 10000
    learning_rate = 1
    x = np.c_[np.ones(len(X)), X]
    w,costHistory = gradientDescent(x,Y,iteration,learning_rate)
    #plot
    plt.figure(2)
    plt.plot(range(iteration),costHistory)
    plt.show()
    plt.figure(3)
    plt.scatter(data_pos[:,0], data_pos[:,1],c='y')
    plt.scatter(data_neg[:,0], data_neg[:,1],c='b')
    x_ = np.linspace(0,1,100)
    y_ = -w[1]/w[2] * x_ - w[0]/w[2]
    plt.plot(x_,y_,'-r')
    plt.show()
    #predict
    x_predict = np.array([1, (45-x_min[0])/x_range[0], (85-x_min[1])/x_range[1]])
    print(predict(w,x_predict))