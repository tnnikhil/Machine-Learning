import csv
import numpy as np
from scipy.stats import zscore
r = csv.reader(open('C:\\ML\\17CS10054_ML_A2\\data\\winequality-red.csv'))
lines = list(r)
for i in range(0,len(lines)):
	a=lines[i][0]
	b=[]
	b=a.split(";")
	lines[i]=b
for i in range(1,len(lines)):
	lines[i]=list(map(float,lines[i]))
for i in range(1,len(lines)):
	if (lines[i][11]<5):
		lines[i][11]=0
	elif(lines[i][11]<7):
		lines[i][11]=1
	else:
		lines[i][11]=2
for j in range(0,11):
	a=[]
	for i in range(1,len(lines)):
		a.append(lines[i][j])
	a=zscore(a)
	min=10000
	max=-10000
	for k in range(0,len(a)):
		if(a[k]<min):
			min=a[k]
		if(a[k]>max):
			max=a[k]
	n=(max-min)/4.0
	for i in range(1,len(lines)):
		if(a[i-1]<min+n):
			lines[i][j]=0
		elif(a[i-1]<min+(2.0)*n):
			lines[i][j]=1
		elif(a[i-1]<min+(3.0)*n):
			lines[i][j]=2
		else:
			lines[i][j]=3
	a*=0
writer = csv.writer(open('./datasetB.csv','w'))
writer.writerows(lines)