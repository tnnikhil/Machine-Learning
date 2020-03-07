from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')
X=[]
Y=[]
x,y=np.genfromtxt('C:\\ML\\test.csv',unpack=True,delimiter=',')
for itr in range(0,200):
    X.append(x[itr+1])
    Y.append(y[itr+1])

a=list(zip(X,Y))
b=sorted(a)
x_new=[x for x,y in b]
y_new=[y for x,y in b]
plt.plot(x_new,y_new)
plt.title('Label vs Feature for testing data')
plt.ylabel('Label')
plt.xlabel('Feature')

plt.show()