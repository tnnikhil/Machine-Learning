from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
C=[0.25,0.5,0.75,1]
plt.plot(C,E1_min)
plt.plot(C,F1_min)
plt.title('Training and Test Errors for various lambdas')
plt.ylabel('Error')
plt.xlabel('lambda value')