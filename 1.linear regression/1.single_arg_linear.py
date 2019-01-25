import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

FILE = 'task1.txt'
# 梯度下降法GD
def gradient_descent(x, y, theta, alpha = 0.01, num_iters = 2000, eps = 1e-4):
    # Initialize some useful values
    m = len(y)
    J_history = np.zeros(num_iters)
    last_theta = theta
    count = 0
    while count < num_iters:
        sum_n = np.zeros(2)
        for i in np.arange(m):
            dif = (np.dot(x[i],theta)-y[i]) * x[i]
            sum_n += dif;
        theta = theta - alpha * sum_n / m
        if  np.linalg.norm(theta - last_theta) < eps:
            return theta, J_history
        else:
            J_history[count] = compute_cost(x, y, last_theta)
            last_theta = theta
            count += 1
    return theta,J_history #迭代次数达到上限

def compute_cost(x,y,theta):
    m = len(y)
    cost = 0;
    for i in np.arange(m):
        cost += (np.dot(x[i],theta) - y[i])**2
    cost = cost/2/m
    return cost
data_raw = np.loadtxt(FILE, delimiter = ',')
x = data_raw[:,0]
y = data_raw[:,1]
total = len(y)

print('exer1:')
print('Scatter diagram:')
plt.ion()
plt.figure(0)
plt.plot(x,y,'ro')
plt.show()

#pic = plt.figure()
#pic0.scatter(x,y)
#pic0.set(xlabel = 'Population of City in 10,000s',
#         ylabel = 'Profit in $10,000s',
#         title = 'exer1')
#plt.show()

print('---------------------------------')
print('gradient_decent:')
x = np.c_[np.ones(total),x] #x扩展一列1
theta = np.zeros(2)
alpha = 0.01 #rate
iterations = 2000 #迭代次数
# Compute and display initial cost
print('Initial cost : ' 
      + str(compute_cost(x, y, theta)) 
      + ' (This value should be about 32.07)')
theta, J_history = gradient_descent(x, y, theta, alpha, iterations)
print('Theta found by gradient descent: ' + str(theta.reshape(2)))

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of {:0.3f} (This value should be about 4519.77)'.format(predict1*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of {:0.3f} (This value should be about 45342.45)'.format(predict2*10000))

plt.figure(0)
plt.plot(x[:,1],y,'ro')
line1, = plt.plot(x[:, 1], np.dot(x, theta), label='Linear Regression')
plt.legend(handles=[line1])
plt.show()
 
#使用scipy库进行线性回归分析
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:,1], y)
print('intercept = %s slope = %s' % (intercept, slope))
