#Decision tree using scikit learn and metrics correpsonding to entire training data
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics 

pima = pd.read_csv("C:\\ML\\17CS10054_ML_A2\\data\\datasetB.csv")
feature_cols=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
X=pima[feature_cols]
y=pima.quality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=1)
clf = DecisionTreeClassifier(min_samples_split=10,criterion='entropy')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print("Accuracy:",100*metrics.accuracy_score(y_train, y_pred))
print("Macro Precision:",100*metrics.precision_score(y_train, y_pred,average='macro'))
print("Macro Recall:",100*metrics.recall_score(y_train, y_pred,average='macro'))