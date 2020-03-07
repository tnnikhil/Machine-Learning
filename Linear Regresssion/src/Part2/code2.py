from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
C=[]
for itr in range(1,10):
    C.append(itr)
plt.plot(C,A)
plt.plot(C,B)
plt.title('Training and Test Errors for various n(s)')
plt.ylabel('Error')
plt.xlabel('order')