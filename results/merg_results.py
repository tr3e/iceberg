import numpy as np 
import pandas as pd 
import os


a = pd.read_csv('iceberg_svm.csv')
b = pd.read_csv('iceberg_rf.csv')

c = []
svm= a['is_iceberg'].astype('float32')
rf = b['is_iceberg'].astype('float32')
for i in xrange(8424):
    if np.abs(svm[i] - rf[i]) <= 0.25:
        c.append(svm[i])
    else:
        c.append(rf[i])

submission=pd.DataFrame({'id':a['id'], 'is_iceberg':c})
submission.to_csv('../results/ice_berg.csv', index=False)