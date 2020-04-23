def cal_nmi(X,a):
    labels=unique(X['label'])
    no_of_data_points_in_each_class=[]
    for label in labels:
        c=0
        for itr in range(0,len(X)):
            if(X['label'][itr]==label):
                c=c+1
        no_of_data_points_in_each_class.append(c)
        
    h_y=0
    for i in range(0,len(no_of_data_points_in_each_class)):
        temp=float(no_of_data_points_in_each_class[i]/len(X))
        h_y=h_y+float(temp*math.log2(temp))
    h_y=-h_y
    #print(h_y)
    
    no_of_data_points_in_each_cluster=[]
    for itr in range(0,len(a)):
        no_of_data_points_in_each_cluster.append(len(a[itr]))
    
    h_c=0
    for i in range(0,len(no_of_data_points_in_each_cluster)):
        temp=float(no_of_data_points_in_each_cluster[i]/len(X))
        h_c=h_c+float(temp*math.log2(temp))
    h_c=-h_c
    #print(h_c)
    
    h_yc=0
    for i in range(0,len(a)):
        cluster=a[i]
        no_of_data_points_in_each_class1=[]
        for label in labels:
            c=0
            for itr in range(0,len(cluster)):
                if(X['label'][cluster[itr]]==label):
                    c=c+1
            no_of_data_points_in_each_class1.append(c)
        for i in range(0,len(no_of_data_points_in_each_class1)):
            temp=float(no_of_data_points_in_each_class1[i]/len(cluster))
            if temp!=0:
                h_yc=h_yc+float(temp*math.log2(temp))
    h_yc=h_yc/8
    h_yc=-h_yc
    #print(h_yc)
    
    nmi=float(float(2*(h_y-h_yc))/float(h_y+h_c))
    return nmi