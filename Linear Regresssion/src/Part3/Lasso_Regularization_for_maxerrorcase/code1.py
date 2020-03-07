j=2
l=0.25
print('Lasso regularization for order=1 and l=0.25')
theta=np.random.randn(j,1)
X=np.zeros((1000,j))
Y=np.zeros((1000,1))
x,y=np.genfromtxt('C:\\ML\\train.csv',unpack=True,delimiter=',')
for itr in range(0,1000):
    for k in range(1,j):
        X[itr][k]=pow(x[itr+1],k)
    Y[itr][0]=y[itr+1]
    X[itr][0]=1
theta,cost_history,theta_history,it=gradient_descent(X,Y,j,theta,0.05,5000000,l)
print('parameters:')
print(theta)
print('Error on Training set:{:0.20f}'.format(cost_history[it]))
E_max.append(cost_history[it])
X1=np.zeros((1000,j))
Y1=np.zeros((1000,1))
x1,y1=np.genfromtxt('C:\\ML\\test.csv',unpack=True,delimiter=',')
for itr in range(0,200):
    for k in range(1,j):
        X1[itr][k]=pow(x1[itr+1],k)
    Y1[itr][0]=y1[itr+1]
    X1[itr][0]=1
print('Test Error:')
F_max.append(cal_cost(theta,X1,Y1,l))
print(F_min[0])
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

j=2
l=0.5
print('Lasso regularization for order=2 and l=0.5')
theta=np.random.randn(j,1)
X=np.zeros((1000,j))
Y=np.zeros((1000,1))
x,y=np.genfromtxt('C:\\ML\\train.csv',unpack=True,delimiter=',')
for itr in range(0,1000):
    for k in range(1,j):
        X[itr][k]=pow(x[itr+1],k)
    Y[itr][0]=y[itr+1]
    X[itr][0]=1
theta,cost_history,theta_history,it=gradient_descent(X,Y,j,theta,0.05,5000000,l)
print('parameters:')
print(theta)
print('Error on Training set:{:0.20f}'.format(cost_history[it]))
E_max.append(cost_history[it])
X1=np.zeros((1000,j))
Y1=np.zeros((1000,1))
x1,y1=np.genfromtxt('C:\\ML\\test.csv',unpack=True,delimiter=',')
for itr in range(0,200):
    for k in range(1,j):
        X1[itr][k]=pow(x1[itr+1],k)
    Y1[itr][0]=y1[itr+1]
    X1[itr][0]=1
print('Test Error:')
F_max.append(cal_cost(theta,X1,Y1,l))
print(F_min[1])
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

j=2
l=0.75
print('Lasso regularization for order=1 and l=0.75')
theta=np.random.randn(j,1)
X=np.zeros((1000,j))
Y=np.zeros((1000,1))
x,y=np.genfromtxt('C:\\ML\\train.csv',unpack=True,delimiter=',')
for itr in range(0,1000):
    for k in range(1,j):
        X[itr][k]=pow(x[itr+1],k)
    Y[itr][0]=y[itr+1]
    X[itr][0]=1
theta,cost_history,theta_history,it=gradient_descent(X,Y,j,theta,0.05,5000000,l)
print('parameters:')
print(theta)
print('Error on Training set:{:0.20f}'.format(cost_history[it]))
E_max.append(cost_history[it])
X1=np.zeros((1000,j))
Y1=np.zeros((1000,1))
x1,y1=np.genfromtxt('C:\\ML\\test.csv',unpack=True,delimiter=',')
for itr in range(0,200):
    for k in range(1,j):
        X1[itr][k]=pow(x1[itr+1],k)
    Y1[itr][0]=y1[itr+1]
    X1[itr][0]=1
print('Test Error:')
F_max.append(cal_cost(theta,X1,Y1,l))
print(F_min[2])
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

j=2
l=1
print('Lasso regularization for order=1 and l=1')
theta=np.random.randn(j,1)
X=np.zeros((1000,j))
Y=np.zeros((1000,1))
x,y=np.genfromtxt('C:\\ML\\train.csv',unpack=True,delimiter=',')
for itr in range(0,1000):
    for k in range(1,j):
        X[itr][k]=pow(x[itr+1],k)
    Y[itr][0]=y[itr+1]
    X[itr][0]=1
theta,cost_history,theta_history,it=gradient_descent(X,Y,j,theta,0.05,5000000,l)
print('parameters:')
print(theta)
print('Error on Training set:{:0.20f}'.format(cost_history[it]))
E_max.append(cost_history[it])
X1=np.zeros((1000,j))
Y1=np.zeros((1000,1))
x1,y1=np.genfromtxt('C:\\ML\\test.csv',unpack=True,delimiter=',')
for itr in range(0,200):
    for k in range(1,j):
        X1[itr][k]=pow(x1[itr+1],k)
    Y1[itr][0]=y1[itr+1]
    X1[itr][0]=1
print('Test Error:')
F_max.append(cal_cost(theta,X1,Y1,l))
print(F_min[3])
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