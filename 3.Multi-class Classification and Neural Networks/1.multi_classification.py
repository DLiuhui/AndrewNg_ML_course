# using logistic regression

import loadData
import numpy as np
import matplotlib.pyplot as plt

FILEPATH = './ex3data1.mat'
SIGMOID = lambda x, w: 1 / (1 + np.e ** (-np.dot(x, w)))
COST_FUNCTION = lambda x, y, w: 1 / len(x) * \
                                (np.dot(np.transpose(-y), np.log(SIGMOID(x, w))) -
                                 np.dot(np.transpose(1 - y), np.log(1 - SIGMOID(x, w))))
DERIVATE_COST_FUNCTION = lambda x, y, w: 1 / len(x) * np.dot(np.transpose(x), SIGMOID(x, w) - y)

def gradientDescent(data, label, para,
                    iteration = 1000,
                    learning_rate = 0.01,
                    eps = 1e-3):
    cost_his = []
    for iter in range(iteration):
        cost_his.append(COST_FUNCTION(data, label, para).item())
        gradient = DERIVATE_COST_FUNCTION(data, label, para)
        new_para = para - learning_rate * gradient 
        if np.max(new_para - para) < eps:
            break
        para = new_para
    return para, cost_his
s

if __name__ == '__main__':
    X, y = loadData.loadData(FILEPATH)
    (n_samples, n_features) = X.shape
    loadData.displayData(X)
    # add x0 to X
    X_ = np.c_[np.ones(n_samples), X]
    w_init = np.zeros((n_features + 1, 1))
    w, cost_his = gradientDescent(X_, y, w_init,
                                  iteration=1000,
                                  learning_rate=0.01)
    print(w)
    print(cost_his)
    plt.figure()
    plt.plot(range(len(cost_his)), cost_his)
    plt.show()



