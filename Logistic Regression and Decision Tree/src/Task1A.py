import csv
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
	if (lines[i][11]<=6):
		lines[i][11]=0
	else:
		lines[i][11]=1
for i in range(0,11):
	min=1000
	max=-1
	for j in range(1,len(lines)):
		if(lines[j][i]<min):
			min=lines[j][i]
		if(lines[j][i]>max):
			max=lines[j][i]
	for j in range(1,len(lines)):
		lines[j][i]=(lines[j][i]-min)/(max-min)
writer = csv.writer(open('./datasetA.csv','w'))
writer.writerows(lines)