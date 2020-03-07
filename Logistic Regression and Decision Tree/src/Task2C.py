import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
m=1599
n=533
r = csv.reader(open('C:\\ML\\17CS10054_ML_A2\\data\\datasetA.csv'))
lines = list(r)
for i in range(1,len(lines)):
    lines[i]=list(map(float,lines[i]))
for i in range(1,len(lines)):
    lines[i]=np.asarray(lines[i])
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
            #print("iterations count:")
            #print(it+1)
            break
    return theta
logisticRegr=LogisticRegression(solver='saga')
X=np.zeros((m,12))
Y=np.zeros((m,1))
for itr in range(1,len(lines)):
    Y[itr-1][0]=lines[itr][11]
    for k in range(1,12):
        X[itr-1][k]=lines[itr][k-1]
    X[itr-1][0]=1
X11=np.zeros((2*n,12))
X12=np.zeros((n,12))
Y11=np.zeros((2*n,1))
Y12=np.zeros((n,1))


for i in range(0,2*n):
    X11[i]=X[i]
    Y11[i]=Y[i]
for i in range(2*n,m):
    X12[i-2*n]=X[i]
    Y12[i-2*n]=Y[i]
logisticRegr.fit(X11,np.ravel(Y11))
predictions = logisticRegr.predict(X12)
tp=0
tn=0
fp=0
fn=0
for i in range(0,n):
    if(predictions[i]==1 and Y12[i]==1):
        tp=tp+1
    if(predictions[i]==0 and Y12[i]==0):
        tn=tn+1
    if(predictions[i]==1 and Y12[i]==0):
        fp=fp+1
    if(predictions[i]==0 and Y12[i]==1):
        fn=fn+1
accuracy2=((tp+tn)/(tp+tn+fp+fn))*100
precision2=(tp/(tp+fp))*100
recall2=(tp/(tp+fn))*100

theta=np.random.randn(12,1)
theta=gradient_descent(X11,Y11,theta,0.05,100000)
predictions=X12.dot(theta)
for i in range(0,n):
    if(predictions[i]>=0.4):
        predictions[i]=1
    else:
        predictions[i]=0
tp=0
tn=0
fp=0
fn=0
for i in range(0,n):
    if(predictions[i]==1 and Y12[i]==1):
        tp=tp+1
    if(predictions[i]==0 and Y12[i]==0):
        tn=tn+1
    if(predictions[i]==1 and Y12[i]==0):
        fp=fp+1
    if(predictions[i]==0 and Y12[i]==1):
        fn=fn+1
accuracy1=((tp+tn)/(tp+tn+fp+fn))*100
precision1=(tp/(tp+fp))*100
recall1=(tp/(tp+fn))*100




for i in range(0,n):
    X11[i]=X[i]
    Y11[i]=Y[i]
for i in range(2*n,m):
    X11[i-n]=X[i]
    Y11[i-n]=Y[i]
for i in range(n,2*n):
    X12[i-n]=X[i]
    Y12[i-n]=Y[i]
logisticRegr.fit(X11,np.ravel(Y11))
predictions = logisticRegr.predict(X12)
tp=0
tn=0
fp=0
fn=0
for i in range(0,n):
    if(predictions[i]==1 and Y12[i]==1):
        tp=tp+1
    if(predictions[i]==0 and Y12[i]==0):
        tn=tn+1
    if(predictions[i]==1 and Y12[i]==0):
        fp=fp+1
    if(predictions[i]==0 and Y12[i]==1):
        fn=fn+1
accuracy2=accuracy2+(((tp+tn)/(tp+tn+fp+fn))*100)
precision2=precision2+((tp/(tp+fp))*100)
recall2=recall2+((tp/(tp+fn))*100)

theta=np.random.randn(12,1)
theta=gradient_descent(X11,Y11,theta,0.05,100000)
predictions=X12.dot(theta)
for i in range(0,n):
    if(predictions[i]>=0.4):
        predictions[i]=1
    else:
        predictions[i]=0
tp=0
tn=0
fp=0
fn=0
for i in range(0,n):
    if(predictions[i]==1 and Y12[i]==1):
        tp=tp+1
    if(predictions[i]==0 and Y12[i]==0):
        tn=tn+1
    if(predictions[i]==1 and Y12[i]==0):
        fp=fp+1
    if(predictions[i]==0 and Y12[i]==1):
        fn=fn+1
accuracy1=accuracy1+(((tp+tn)/(tp+tn+fp+fn))*100)
precision1=precision1+((tp/(tp+fp))*100)
recall1=recall1+((tp/(tp+fn))*100)





for i in range(n,m):
    X11[i-n]=X[i]
    Y11[i-n]=Y[i]
for i in range(0,n):
    X12[i]=X[i]
    Y12[i]=Y[i]
logisticRegr.fit(X11,np.ravel(Y11))
predictions = logisticRegr.predict(X12)
tp=0
tn=0
fp=0
fn=0
for i in range(0,n):
    if(predictions[i]==1 and Y12[i]==1):
        tp=tp+1
    if(predictions[i]==0 and Y12[i]==0):
        tn=tn+1
    if(predictions[i]==1 and Y12[i]==0):
        fp=fp+1
    if(predictions[i]==0 and Y12[i]==1):
        fn=fn+1
accuracy2=accuracy2+((tp+tn)/(tp+tn+fp+fn))*100
precision2=precision2+(tp/(tp+fp))*100
recall2=recall2+(tp/(tp+fn))*100

theta=np.random.randn(12,1)
theta=gradient_descent(X11,Y11,theta,0.05,100000)
predictions=X12.dot(theta)
for i in range(0,n):
    if(predictions[i]>=0.4):
        predictions[i]=1
    else:
        predictions[i]=0
tp=0
tn=0
fp=0
fn=0
for i in range(0,n):
    if(predictions[i]==1 and Y12[i]==1):
        tp=tp+1
    if(predictions[i]==0 and Y12[i]==0):
        tn=tn+1
    if(predictions[i]==1 and Y12[i]==0):
        fp=fp+1
    if(predictions[i]==0 and Y12[i]==1):
        fn=fn+1
accuracy1=accuracy1+(((tp+tn)/(tp+tn+fp+fn))*100)
precision1=precision1+((tp/(tp+fp))*100)
recall1=recall1+((tp/(tp+fn))*100)

accuracy1=accuracy1/3
precision1=precision1/3
recall1=recall1/3
accuracy2=accuracy2/3
precision2=precision2/3
recall2=recall2/3

print("For Custom Logistic Regression:")
print("Average Test Accuracy is ")
print(accuracy1)
print("\nAverage Test Precision is")
print(precision1)
print("\nAverage Test Recall is")
print(recall1)
print("\n\n\nFor scikit Learn Logistic Regression:")
print("Average Test Accuracy is ")
print(accuracy2)
print("\nAverage Test Precision is")
print(precision2)
print("\nAverage Test Recall is")
print(recall2)