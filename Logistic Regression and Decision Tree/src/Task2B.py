import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
m=1599
r = csv.reader(open('C:\\ML\\17CS10054_ML_A2\\data\\datasetA.csv'))
lines = list(r)
for i in range(1,len(lines)):
    lines[i]=list(map(float,lines[i]))
for i in range(1,len(lines)):
    lines[i]=np.asarray(lines[i])
logisticRegr=LogisticRegression(solver='saga')
X=np.zeros((m,12))
Y=np.zeros((m,1))
for itr in range(1,len(lines)):
    Y[itr-1][0]=lines[itr][11]
    for k in range(1,12):
        X[itr-1][k]=lines[itr][k-1]
    X[itr-1][0]=1
logisticRegr.fit(X,np.ravel(Y))
predictions = logisticRegr.predict(X)
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