class K_Means:
    def __init__(self,k=8,tol=0.0001,max_iter=600):
        self.k=k
        self.tol=tol
        self.max_iter=max_iter
        
    def fit(self,data):
        
        self.centroids = []
        
        for i in range(self.k):
            self.centroids.append(data[i])
        
        for i in range(self.max_iter):
            self.classifications= {}
            
            for i in range(self.k):
                self.classifications[i]=[]
                
            for featureset in data:
                distances=[]
                for centroid in self.centroids:
                    #print(featureset)
                    #print(centroid)
                    temp=np.dot(featureset,centroid)
                    #print(temp)
                    distances.append(float(math.exp(-1*temp)))
                classification=distances.index(min(distances))
                self.classifications[classification].append(featureset)
                
            prev_centroids=list(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification]=np.average(self.classifications[classification],axis=0)
                
            optimized=True
            '''
            for c in range(self.k):
                original_centroid=prev_centroids[c]
                current_centroid=self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100) > self.tol:
                    optimized=False
                    
            if optimized:
                print("iteration count:")
                print(i)
                break
            '''
    def predict(self,data):
        distances=[float(math.exp(-1*np.dot(np.array(data),np.array(centroid)))) for centroid in self.centroids]
        classification=distances.index(min(distances))
        return classification