#import all necessary modules
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#the following describes all  necessary activation functions and their derivatives
def sigm(m):
    return 1/(1+np.exp(-m))

def sigm_dif(m):
    return sigm(m)*(1-sigm(m))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def ReLU(m):
    return m * (m > 0)

def ReLU_diff(m):
    return 1. * (m > 0)

#"preprocess" module takes the "seeds" dataset and converts the output into one hot encoding and splits data into training and testing data and stores back them as csv files. 
def preprocess(filename):
    column_names=['atr1','atr2','atr3','atr4','atr5','atr6','atr7','class']
    X=pd.read_csv(filename,names=column_names,header=None)

    Y=X.apply(zscore)
    del Y['class']
    Y['out1']=0
    Y['out2']=0
    Y['out3']=0
    for itr in range(0,len(X)):
        if (X['class'][itr]==1):
            Y['out1'][itr]=1
        if (X['class'][itr]==2):
            Y['out2'][itr]=1
        if (X['class'][itr]==3):
            Y['out3'][itr]=1


    train, test = train_test_split(Y, test_size=0.2)
    train.to_csv('./data/train_data.csv', sep=',',index=False)
    test.to_csv('./data/test_data.csv', sep=',',index=False)

#loads train and test data and splits into batches of sie 32 and returns the batches
def data_loader(trainfile_path):
    train=pd.read_csv(trainfile_path).to_numpy()
    np.random.shuffle(train)
    batches=np.split(train,[32,64,96,128,160])
    return batches

#'n' is number of features
#'r' is number of result classes
#'l' is a list detailing hidden layer architecture.Eg: If 3 hidden layers and no of neurons are 30,40,50 then l=[30,40,50]

#initializes weights and stores as numpy arrays
def weight_initializer(n,l,r):
    res = []
    
    a=np.zeros((l[0],n+1))
    for x in range(0, a.shape[0]):
        for y in range(0, a.shape[1]):
            a[x,y]=np.random.uniform(-1,1)
    res.append(a)
    
    for i in range(len(l)-1):
        a=np.empty((l[i+1],l[i]+1))
        for x in range(0, a.shape[0]):
            for y in range(0, a.shape[1]):
                a[x,y]=np.random.uniform(-1,1)
        res.append(a)
    
    a=np.empty((r,l[-1]+1))
    for x in range(0, a.shape[0]):
        for y in range(0, a.shape[1]):
            a[x,y]=np.random.uniform(-1,1)
    res.append(a)
    return res

#'x' initializer i.e, to store output of each neuron
def x_initializer(n,l,r):
    res=[]
    
    a=np.ones(n+1)
    res.append(a)
    
    for i in range(0,len(l)):
        a=np.ones(l[i]+1)
        res.append(a)
    
    a=np.ones(r)
    res.append(a)
    return res

#'delta' initializer to store delta of each neuron while backpropagating
def delta_initializer(n,l,r):
    res=[]
    
    for i in range(0,len(l)):
        a=np.zeros(l[i])
        res.append(a)
        
    a=np.zeros(r)
    res.append(a)
    return res

#'gradient' initializer to store gradients of each parameter(weight) 
def gradient_initializer(n,l,r):
    res = []
    
    a=np.zeros((l[0],n+1))
    res.append(a)
    
    for i in range(len(l)-1):
        a=np.zeros((l[i+1],l[i]+1))
        res.append(a)
    
    a=np.zeros((r,l[-1]+1))
    res.append(a)
    return res

#initializes weights and stores as numpy arrays
def weight_initializer1(n,l,r):
    res = []
    res.append(np.sqrt(1/(n+1))*np.random.randn(l[0],n+1))
    for i in range(len(l)-1):
        res.append(np.sqrt(1/(l[i]+1))*np.random.randn(l[i+1],l[i]+1))
    res.append(np.sqrt(1/(l[-1]+1))*np.random.randn(r,l[-1]+1))
    return res

#this module calculates the output of each neuron taking one sample from the dataset and updating corresponding final probabilities 
def forward(weights,x,n,l,r,act_fun_hidden,act_fun_output):
    temp=len(l)
    for i in range(0,temp):
        c=np.dot(weights[i],x[i])
        c=act_fun_hidden(c)
        x[i+1]=np.hstack((1.0,c))
    
    c=np.dot(weights[temp],x[temp])
    x[-1]=act_fun_output(c)
    return x

#this module updates all the weights during backpropagation taking help of gradients calculated from "cal_gradients" module
def backward(weights,gradients,alpha):
    for i in range(0,len(weights)):
        weights[i]=weights[i]-alpha*gradients[i]
    return weights

#to calculate gradients after each "forward" run of a sample
def cal_gradients(weights,x,n,l,r,delta,alpha):
    res=[]
    for i in reversed(range(0,len(l))):
        a=np.delete(x[i+1],0)
        a=1-a*a
        b=np.delete(weights[i+1],0,axis=1)
        c=np.dot(b.T,delta[i+1])
        delta[i]=c*a
    
    for i in range(0,len(l)+1):
        a=delta[i]
        a=a.reshape(delta[i].shape[0],1)
        b=x[i]
        b=b.reshape(1,x[i].shape[0])
        c=np.dot(a,b)
        res.append(c)
    return res

#takes a batch and calculates the probabilities and eventually number of correct outputs and stores gradients after each run to update weights 
def predict(weights,x,n,l,r,delta,alpha,act_fun_hidden,act_fun_output,df):
    count=0
    res_pred=1
    res_crct=1
    gradients=gradient_initializer(n,l,r)
    for i in range(0,df.shape[0]):
        a=df[i]
        b=[a[-3],a[-2],a[-1]]
        if(b[0]==1):
            res_crct=1
        if(b[1]==1):
            res_crct=2
        if(b[2]==1):
            res_crct=3
        a=np.delete(a,-1)
        a=np.delete(a,-1)
        a=np.delete(a,-1)
        a=np.hstack((1.0,a))
        x[0]=a
        x=forward(weights,x,n,l,r,act_fun_hidden,act_fun_output)
        a=x[-1]
        if(a[0]>=a[1] and a[0]>=a[2]):
            res_pred=1
        if(a[1]>=a[0] and a[1]>=a[2]):
            res_pred=2
        if(a[2]>=a[1] and a[2]>=a[0]):
            res_pred=3
        if(res_pred==res_crct):
            count=count+1
        temp=np.array([a[0]-b[0],a[1]-b[1],a[2]-b[2]])
        delta[-1]=temp
        res=cal_gradients(weights,x,n,l,r,delta,alpha)
        for i in range(0,len(gradients)):
            gradients[i]=gradients[i]+res[i]
    
    for i in range(0,len(gradients)):
        gradients[i]=gradients[i]/df.shape[0]
    return count,x,gradients

preprocess('./data/seeds.csv')
batches=data_loader('./data/train_data.csv')
train=pd.read_csv("./data/train_data.csv").to_numpy()
test=pd.read_csv("./data/test_data.csv").to_numpy()

#part1A
#all initializations
weights=weight_initializer(7,[32],3)
x=x_initializer(7,[32],3)
delta=delta_initializer(7,[32],3)
gradients=gradient_initializer(7,[32],3)

epochs=200
training_acc,testing_acc=[],[]

#Mini batch SGD Loop to implement backpropagation algorithm
for i in range(0,epochs):
    
    if(i%10 ==0 and i!=0):
        train_acc,x,gradients=predict(weights,x,7,[32],3,delta,0.01,sigm,softmax,train)
        train_acc=train_acc/train.shape[0]
        training_acc.append(train_acc)
        
        test_acc,x,gradients=predict(weights,x,7,[32],3,delta,0.01,sigm,softmax,test)
        test_acc=test_acc/test.shape[0]
        testing_acc.append(test_acc)
    
    for j in range(0,len(batches)):
        count,x,gradients=predict(weights,x,7,[32],3,delta,0.01,sigm,softmax,batches[j])
        weights=backward(weights,gradients,0.01)
        
        
final_train_acc,x,gradients=predict(weights,x,7,[32],3,delta,0.01,sigm,softmax,train)
final_train_acc=final_train_acc/train.shape[0]

final_test_acc,x,gradients=predict(weights,x,7,[32],3,delta,0.01,sigm,softmax,test)
final_test_acc=final_test_acc/test.shape[0]

print('PART1A:')
b=[]
for i in range(1,21):
    b.append(10*i)

#store training and testing accuracies after every 10 epochs 
training_acc.append(final_train_acc)
testing_acc.append(final_test_acc)
figure(figsize=(12,6))
plt.plot(b,training_acc, color='red', linewidth=2, label='TrainAcc')
plt.plot(b,testing_acc, color='green', linewidth=2, label="TestAcc")
plt.xlabel("No. of epochs completed")
plt.ylabel("Accuracy")
plt.xticks(b)
plt.legend()
plt.show()

#print final accuracies
print('Final Training Accuracy:')
print(final_train_acc)
print('Final Testing Accuracy:')
print(final_test_acc)
print()


#part1B
#all initializations
weights=weight_initializer1(7,[64,32],3)
x=x_initializer(7,[64,32],3)
delta=delta_initializer(7,[64,32],3)
gradients=gradient_initializer(7,[64,32],3)

epochs=200
alpha=0.01
training_acc,testing_acc=[],[]

#Mini batch SGD Loop to implement backpropagation algo
for i in range(0,epochs):
    
    if(i%10 ==0 and i!=0):
        train_acc,x,gradients=predict(weights,x,7,[64,32],3,delta,alpha,ReLU,softmax,train)
        train_acc=train_acc/train.shape[0]
        training_acc.append(train_acc)
        
        test_acc,x,gradients=predict(weights,x,7,[64,32],3,delta,alpha,ReLU,softmax,test)
        test_acc=test_acc/test.shape[0]
        testing_acc.append(test_acc)
    
    for j in range(0,len(batches)):
        count,x,gradients=predict(weights,x,7,[64,32],3,delta,alpha,ReLU,softmax,batches[j])
        weights=backward(weights,gradients,alpha)
        
final_train_acc,x,gradients=predict(weights,x,7,[64,32],3,delta,alpha,ReLU,softmax,train)
final_train_acc=final_train_acc/train.shape[0]

final_test_acc,x,gradients=predict(weights,x,7,[64,32],3,delta,alpha,ReLU,softmax,test)
final_test_acc=final_test_acc/test.shape[0]

print('PART1B:')
b=[]
for i in range(1,21):
    b.append(10*i)

#to store accuracies after every 10 epochs
training_acc.append(final_train_acc)
testing_acc.append(final_test_acc)
figure(figsize=(12,6))
plt.plot(b,training_acc, color='red', linewidth=2, label='TrainAcc')
plt.plot(b,testing_acc, color='green', linewidth=2, label="TestAcc")
plt.xlabel("No. of epochs completed")
plt.ylabel("Accuracy")
plt.xticks(b)
plt.legend()
plt.show()

#print final training and testing accuracies
print('Final Training Accuracy:')
print(final_train_acc)
print('Final Testing Accuracy:')
print(final_test_acc)
print()