import pandas as pd
from sklearn.metrics import SCORERS
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import metrics 

pima = pd.read_csv("C:\\ML\\17CS10054_ML_A2\\data\\datasetB.csv")
feature_cols=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
X=pima[feature_cols]
y=pima.quality
clf = DecisionTreeClassifier(min_samples_split=10,criterion='entropy')
scores = cross_validate(clf, X, y, cv=3,scoring=('accuracy','precision_macro','recall_macro'),return_train_score=True)
a=(scores['test_accuracy'])
b=(scores['test_precision_macro'])
c=(scores['test_recall_macro'])
print('For Scikit Learn Decision Tree Classifier and 3 fold cross validation:')
print('Average Test Accuracy is ',(a[0]+a[1]+a[2])/3*100)
print('Average Test Macro Precision is ',(b[0]+b[1]+b[2])/3*100)
print('Average Test Macro Recall is ',(c[0]+c[1]+c[2])/3*100)