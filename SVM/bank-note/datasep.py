import numpy as np

train_data = np.loadtxt('/Users/wjy/Downloads/bank-note/train.csv',delimiter=',')
test_data = np.loadtxt('/Users/wjy/Downloads/bank-note/test.csv',delimiter=',')

train_data[:,-1] = 2*train_data[:,-1] - 1
test_data[:,-1] = 2*test_data[:,-1] - 1

np.savetxt('/Users/wjy/Downloads/bank-note/train.txt',train_data,fmt='%3f')
np.savetxt('/Users/wjy/Downloads/bank-note/test.txt',test_data,fmt='%3f')


