import csv
import numpy as np
r = csv.reader(open('C:\\ML\\17CS10054_ML_A2\\data\\datasetA.csv'))
lines = list(r)
for i in range(1,len(lines)):
    lines[i]=list(map(float,lines[i]))
for i in range(1,len(lines)):
    lines[i]=np.asarray(lines[i])
m=1599
def cal_cost(theta,X,Y):
    m=len(Y)
    cost=0
    predictions=X.dot(theta)
    for i in  range(0,m):
        a = 1/(1 + np.exp(-predictions[i]))
        temp=Y[i]*(np.log(a))+(1-Y[i])*(np.log(1-a))
        cost=cost + temp
    cost=-cost
    cost=cost/m
    return cost

def gradient_descent(X,Y,theta,learning_rate,iterations):
    m=len(Y)
    cost_history=np.zeros(iterations)
    theta_history=np.zeros((iterations,12))
    for it in range(iterations):
        prediction=np.dot(X,theta)
        theta=theta-(1/m)*learning_rate*(X.T.dot(prediction-Y))
        theta_history[it,:]=theta.T
        cost_history[it]=cal_cost(theta,X,Y)
        if((cost_history[it-1]-cost_history[it])<0.00000001 and it>0):
            print("iterations count:")
            print(it+1)
            break
    return theta

theta=np.random.randn(12,1)
X=np.zeros((m,12))
Y=np.zeros((m,1))
for itr in range(1,len(lines)):
    Y[itr-1][0]=lines[itr][11]
    for k in range(1,12):
        X[itr-1][k]=lines[itr][k-1]
    X[itr-1][0]=1
theta=gradient_descent(X,Y,theta,0.05,100000)
print("Final Theta values:")
print(theta)
predictions=X.dot(theta)
for i in range(0,m):
    if(predictions[i]>=0.5):
        predictions[i]=1
    else:
        predictions[i]=0

tp=0
tn=0
fp=0
fn=0
for i in range(0,m):
    if(predictions[i]==1 and Y[i]==1):
        tp=tp+1
    if(predictions[i]==0 and Y[i]==0):
        tn=tn+1
    if(predictions[i]==1 and Y[i]==0):
        fp=fp+1
    if(predictions[i]==0 and Y[i]==1):
        fn=fn+1
print("accuracy is ")
print( ((tp+tn)/(tp+tn+fp+fn))*100 )