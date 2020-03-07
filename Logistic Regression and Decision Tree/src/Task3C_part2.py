import csv
import numpy as np
r = csv.reader(open('C:\\ML\\17CS10054_ML_A2\\data\\datasetB.csv'))
training_data = list(r)
for i in range(1,len(training_data)):
    training_data[i]=list(map(float,training_data[i]))
training_data.pop(0)
a=[]
b=[]
c=[]
m=1599
n=533
X1=[]
X2=[]
def unique_vals(rows, col):
    return set([row[col] for row in rows])


def class_counts(rows):
    counts = {}  
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_split(rows):
    best_gain = 0  
    best_question = None 
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  

    for col in range(n_features):  

        values = set([row[col] for row in rows]) 

        for val in values:  

            question = Question(col, val)

            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)
    
    if len(rows)<10:
        return Leaf(rows)
    
    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)

def classify(row, node):
    
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    Keymax = max(probs, key=probs.get) 
    return Keymax

if __name__ == '__main__':
    count=0
    a1=0
    a2=0
    a3=0
    a4=0
    a5=0
    a6=0
    a7=0
    a8=0
    a9=0
    m=len(training_data)
    for i in range(0,2*n):
        X1.append(training_data[i])
    for i in range(2*n,m):
        X2.append(training_data[i])
    my_tree = build_tree(X1)
   
    for row in X2:
        if(row[-1]==print_leaf(classify(row, my_tree))):
            count=count+1
    a.append(count/n*100)
    for row in X2:
        if(row[-1]==0 and print_leaf(classify(row, my_tree))==0):
            a1=a1+1
        if(row[-1]==0 and print_leaf(classify(row, my_tree))==1):
            a2=a2+1
        if(row[-1]==0 and print_leaf(classify(row, my_tree))==2):
            a3=a3+1
        if(row[-1]==1 and print_leaf(classify(row, my_tree))==0):
            a4=a4+1
        if(row[-1]==1 and print_leaf(classify(row, my_tree))==1):
            a5=a5+1
        if(row[-1]==1 and print_leaf(classify(row, my_tree))==2):
            a6=a6+1
        if(row[-1]==2 and print_leaf(classify(row, my_tree))==0):
            a7=a7+1
        if(row[-1]==2 and print_leaf(classify(row, my_tree))==1):
            a8=a8+1
        if(row[-1]==2 and print_leaf(classify(row, my_tree))==2):
            a9=a9+1
    temp1=a1/(a1+a4+a7)*100
    temp2=a5/(a2+a5+a8)*100
    temp3=a9/(a3+a6+a9)*100
    temp=(temp1+temp2+temp3)/3
    b.append(temp)
    temp1=a1/(a1+a2+a3)*100
    temp2=a5/(a4+a5+a6)*100
    temp3=a9/(a7+a8+a9)*100
    temp=(temp1+temp2+temp3)/3
    c.append(temp)
    X1.clear()
    X2.clear()
    
    count=0
    a1=0
    a2=0
    a3=0
    a4=0
    a5=0
    a6=0
    a7=0
    a8=0
    a9=0
    m=len(training_data)
    for i in range(0,n):
        X2.append(training_data[i])
    for i in range(n,m):
        X1.append(training_data[i])
    my_tree = build_tree(X1)
   
    for row in X2:
        if(row[-1]==print_leaf(classify(row, my_tree))):
            count=count+1
    a.append(count/n*100)
    for row in X2:
        if(row[-1]==0 and print_leaf(classify(row, my_tree))==0):
            a1=a1+1
        if(row[-1]==0 and print_leaf(classify(row, my_tree))==1):
            a2=a2+1
        if(row[-1]==0 and print_leaf(classify(row, my_tree))==2):
            a3=a3+1
        if(row[-1]==1 and print_leaf(classify(row, my_tree))==0):
            a4=a4+1
        if(row[-1]==1 and print_leaf(classify(row, my_tree))==1):
            a5=a5+1
        if(row[-1]==1 and print_leaf(classify(row, my_tree))==2):
            a6=a6+1
        if(row[-1]==2 and print_leaf(classify(row, my_tree))==0):
            a7=a7+1
        if(row[-1]==2 and print_leaf(classify(row, my_tree))==1):
            a8=a8+1
        if(row[-1]==2 and print_leaf(classify(row, my_tree))==2):
            a9=a9+1
    temp1=a1/(a1+a4+a7)*100
    temp2=a5/(a2+a5+a8)*100
    temp3=a9/(a3+a6+a9)*100
    temp=(temp1+temp2+temp3)/3
    b.append(temp)
    temp1=a1/(a1+a2+a3)*100
    temp2=a5/(a4+a5+a6)*100
    temp3=a9/(a7+a8+a9)*100
    temp=(temp1+temp2+temp3)/3
    c.append(temp)
    X1.clear()
    X2.clear()
    
    count=0
    a1=0
    a2=0
    a3=0
    a4=0
    a5=0
    a6=0
    a7=0
    a8=0
    a9=0
    m=len(training_data)
    for i in range(0,n):
        X1.append(training_data[i])
    for i in range(2*n,m):
        X1.append(training_data[i])
    for i in range(n,2*n):
        X2.append(training_data[i])
    my_tree = build_tree(X1)
   
    for row in X2:
        if(row[-1]==print_leaf(classify(row, my_tree))):
            count=count+1
    a.append(count/n*100)
    for row in X2:
        if(row[-1]==0 and print_leaf(classify(row, my_tree))==0):
            a1=a1+1
        if(row[-1]==0 and print_leaf(classify(row, my_tree))==1):
            a2=a2+1
        if(row[-1]==0 and print_leaf(classify(row, my_tree))==2):
            a3=a3+1
        if(row[-1]==1 and print_leaf(classify(row, my_tree))==0):
            a4=a4+1
        if(row[-1]==1 and print_leaf(classify(row, my_tree))==1):
            a5=a5+1
        if(row[-1]==1 and print_leaf(classify(row, my_tree))==2):
            a6=a6+1
        if(row[-1]==2 and print_leaf(classify(row, my_tree))==0):
            a7=a7+1
        if(row[-1]==2 and print_leaf(classify(row, my_tree))==1):
            a8=a8+1
        if(row[-1]==2 and print_leaf(classify(row, my_tree))==2):
            a9=a9+1
    temp1=a1/(a1+a4+a7)*100
    temp2=a5/(a2+a5+a8)*100
    temp3=a9/(a3+a6+a9)*100
    temp=(temp1+temp2+temp3)/3
    b.append(temp)
    temp1=a1/(a1+a2+a3)*100
    temp2=a5/(a4+a5+a6)*100
    temp3=a9/(a7+a8+a9)*100
    temp=(temp1+temp2+temp3)/3
    c.append(temp)
    print('For Custom  Decision Tree Classifier and 3 fold cross validation:')
    print('Average Test Accuracy is ',(a[0]+a[1]+a[2])/3)
    print('Average Test Macro Precision is ',(b[0]+b[1]+b[2])/3)
    print('Average Test Macro Recall is ',(c[0]+c[1]+c[2])/3)