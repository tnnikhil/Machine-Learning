from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
A=[]
B=[]
def cal_cost(theta,X,Y):
    m=len(Y)
    predictions=X.dot(theta)
    cost=(np.sum(np.square(predictions-Y)))/(2*m)
    return cost

def gradient_descent(X,Y,j,theta,learning_rate,iterations):
    m=len(Y)
    cost_history=np.zeros(iterations)
    theta_history=np.zeros((iterations,j))
    for it in range(iterations):
        prediction=np.dot(X,theta)
        theta=theta-(1/m)*learning_rate*(X.T.dot(prediction-Y))
        theta_history[it,:]=theta.T
        cost_history[it]=cal_cost(theta,X,Y)
        if((cost_history[it-1]-cost_history[it])<0.0000001 and it>0):
            print('iterations count:')
            print(it+1)
            break
    return theta,cost_history,theta_history,it

for j in range(2,11):
    print('FOR ORDER:')
    print(j-1)
    print('\n')
    theta=np.random.randn(j,1)
    X=np.zeros((1000,j))
    Y=np.zeros((1000,1))
    x,y=np.genfromtxt('C:\\ML\\train.csv',unpack=True,delimiter=',')
    for itr in range(0,1000):
        for k in range(1,j):
            X[itr][k]=pow(x[itr+1],k)
        Y[itr][0]=y[itr+1]
        X[itr][0]=1
    theta,cost_history,theta_history,it=gradient_descent(X,Y,j,theta,0.05,5000000)
    print('parameters:')
    print(theta)
    print('Error on Training set:{:0.20f}'.format(cost_history[it]))
    A.append(cost_history[it])
    X1=np.zeros((1000,j))
    Y1=np.zeros((1000,1))
    x1,y1=np.genfromtxt('C:\\ML\\test.csv',unpack=True,delimiter=',')
    for itr in range(0,200):
        for k in range(1,j):
            X1[itr][k]=pow(x1[itr+1],k)
        Y1[itr][0]=y1[itr+1]
        X1[itr][0]=1
    print('Test Error:')
    B.append(cal_cost(theta,X1,Y1))
    print(B[j-2])
    print('\n')

    predictions=X1.dot(theta)
    Z=[]
    for itr in range(0,200):
        Z.append(x1[itr+1])
    a=list(zip(Z,predictions))
    b=sorted(a)
    x_new=[x for x,y in b]
    y_new=[y for x,y in b]
    plt.plot(x_new,y_new)
    plt.title('Predicted values of Labels for Test Data')
    plt.ylabel('Predicted Label')
    plt.xlabel('Feature')

    plt.show()
    