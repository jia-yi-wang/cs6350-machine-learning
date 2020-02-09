import numpy as np
import csv

data = []
with open ('/Users/wjy/Downloads/bank/test.csv','r') as f:
    for line in f :
        terms = line.strip().split(',')
        data.append(terms)

d1 = np.array([int(i[0]) for i in data])
d2 = np.array([int(i[5]) for i in data])
d3 = np.array([int(i[9]) for i in data])
d4 = np.array([int(i[11]) for i in data])
d5 = np.array([int(i[12]) for i in data])
d6 = np.array([int(i[13]) for i in data])
d7 = np.array([int(i[14]) for i in data])
m1 = np.median(d1)
m2 = np.median(d2)
m3 = np.median(d3)
m4 = np.median(d4)
m5 = np.median(d5)
m6 = np.median(d6)
m7 = np.median(d7)

print(m7)

for i in range(0,len(data)):
    data[i][0] = int(d1[i]>m1)
    data[i][5] = int(d2[i]>m2)
    data[i][9] = int(d3[i]>m3)
    data[i][11] = int(d4[i]>m4)
    data[i][12] = int(d5[i]>m5)
    data[i][13] = int(d6[i]>m6)
    data[i][14] = int(d7[i]>m7)


with open("processed_test.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerows(data)
