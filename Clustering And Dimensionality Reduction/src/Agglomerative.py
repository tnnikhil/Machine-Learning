cl = []
for i in range(0,len(Y)):
    cl.append([i])
dis_mat = np.zeros((len(Y),len(Y)))
for i in range(0,len(Y)):
    for j in range(0,len(Y)):
        
        if(i==j):
            dis_mat[i][j] = 1000.0
        else:
            dis_mat[i][j] = math.exp(-np.dot(Y[i],Y[j]))


while(len(cl)>=9):
    x,y = np.unravel_index(np.argmin(dis_mat,axis=None), dis_mat.shape)
    
    if(x<y):
        dis_mat[x] = np.where(dis_mat[x]<dis_mat[y],dis_mat[x],dis_mat[y])
        temp = dis_mat[x]
        
        dis_mat = dis_mat.transpose()
        dis_mat[x] = temp

        dis_mat = dis_mat.transpose()
        
        dis_mat = np.delete(dis_mat,y,0) 
        dis_mat = np.delete(dis_mat,y,1) 
        dis_mat[x][x] = 10000
        cl[x] = cl[x] + cl[y]
        cl.pop(y)
    else:
        dis_mat[y] = np.where(dis_mat[y]<dis_mat[x],dis_mat[y],dis_mat[x])
        temp = dis_mat[y]
        
        dis_mat = dis_mat.transpose()
        dis_mat[y] = temp

        dis_mat = dis_mat.transpose()
        
        dis_mat = np.delete(dis_mat,x,0)
        dis_mat = np.delete(dis_mat,x,1)
        dis_mat[y][y] = 10000
        cl[y] = cl[y] + cl[x]
        cl.pop(x)       

for i in range(0,8):
    cl[i].sort()

cl.sort()
print(cl)