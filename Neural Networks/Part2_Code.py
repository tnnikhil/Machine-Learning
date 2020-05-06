#import all necessary modules
from sklearn.neural_network import MLPClassifier 
import pandas as pd
import numpy as np

#load corresponding "Train and Test Data" to numpy arrays
train=pd.read_csv("./data/train_data.csv").to_numpy()
test=pd.read_csv("./data/test_data.csv").to_numpy()

#split the loaded train numpy array into inputs(features) and outputs(one hot encodings) accordingly
train1=np.delete(train,[7,8,9],1)
y_train1=np.delete(train,[0,1,2,3,4,5,6],1)

#split the loaded test numpy array into inputs(features) and outputs(one hot encodings) accordingly
test1=np.delete(test,[7,8,9],1)
y_test1=np.delete(test,[0,1,2,3,4,5,6],1)

#use MLPClassifier with Part1A Specifications
clf=MLPClassifier(max_iter=200,learning_rate_init=0.01,solver='sgd',batch_size=32,activation='logistic',hidden_layer_sizes=(32,),random_state=1)
clf.fit(train1,y_train1)

#print results
print("Part 2 Specification 1A :")
print('Final Training Accuracy:')
print(clf.score(train1,y_train1,sample_weight=None))
print('Final Testing Accuracy:')
print(clf.score(test1,y_test1,sample_weight=None))
print()

#use MLPClassifier with Part1B Specifications
clf=MLPClassifier(max_iter=200,learning_rate_init=0.01,solver='sgd',batch_size=32,activation='relu',hidden_layer_sizes=(64,32,),random_state=1)
clf.fit(train1,y_train1)

#print results
print("Part 2 Specification 1B :")
print('Final Training Accuracy:')
print(clf.score(train1,y_train1,sample_weight=None))
print('Final Testing Accuracy:')
print(clf.score(test1,y_test1,sample_weight=None))