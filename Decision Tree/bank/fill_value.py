import numpy as np
import csv

V = [["admin.","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
     ["secondary","primary","tertiary"],
     ["telephone","cellular"],
     ["other","failure","success"]]

def find_major_value(values,f):
    d = len(values)
    l = len(f)
    count = np.zeros(d)
    for i in range(0,l):
        if f[i] != 'unknown':
            ind = values.index(f[i])
            count[ind] = count[ind] + 1
    im = np.argmax(count)
    return values[im]
    



data = []
with open ('/Users/wjy/Documents/processed_train.csv','r') as f:
    for line in f :
        terms = line.strip().split(',')
        data.append(terms)

ind = np.array([1,3,8,15])
mv = []
for i in range(0,4):
    f = [ii[ind[i]] for ii in data]
    mv.append(find_major_value(V[i],f))


data1 = []
with open ('/Users/wjy/Documents/processed_test.csv','r') as f:
    for line in f :
        terms = line.strip().split(',')
        data1.append(terms)


for i in range(0,len(data1)):
    for j in range(0,4):
        if data1[i][ind[j]] == 'unknown':
            data1[i][ind[j]] = mv[j]

with open("fill_value_test.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerows(data1)
                   
    
