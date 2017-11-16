import os
import numpy as np 
import pandas as pd 
import sys
sys.path.append("..") 
import preprocess as pre
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from keras.models import load_model, Model
from sklearn.externals import joblib
from sklearn.utils import shuffle


data_folder = '/home/itsuki/workspace/iceberg/data'

X_train = pre.load_features('../feature/features_train1.npy')
y_train = pre.load_features('../feature/labels_train1.npy')
shuffle(X_train, y_train, random_state=0)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1314)
for i in xrange(2, 100):
    featrues_file = '../feature/features_train' + str(i) + '.npy'
    labels_file = '../feature/labels_train' + str(i) + '.npy'

    if os.path.exists(featrues_file):
        new_X = pre.load_features(featrues_file)
        new_y = pre.load_features(labels_file)
        shuffle(new_X, new_y, random_state=0)

        # new_X, new_X_val, new_y, new_y_val = train_test_split(new_X, new_y, test_size=0.2, random_state=1314)

        X_train = np.concatenate((X_train, new_X), axis=0).astype('float32')
        y_train = np.concatenate((y_train, new_y), axis=0).astype('float32')

        # X_val = np.concatenate((X_val, new_X_val), axis=0).astype('float32')
        # y_val = np.concatenate((y_val, new_y_val), axis=0).astype('float32')
    else:
        print 'no such file:', featrues_file
        break

print X_train.shape, y_train.shape


clf = SVC(C=80, probability=False,verbose=True)
clf.fit(X_train, y_train)
# print metrics.log_loss(y_train, clf.predict(X_train)), metrics.log_loss(y_val, clf.predict(X_val))
# print c, clf.score(X_val, y_val)

test_df = pre.load_data('test.json')
features_test = pre.load_features('../feature/features_test.npy')

results = clf.predict(features_test)
print results

submission=pd.DataFrame({'id':test_df['id'], 'is_iceberg':results})
submission.to_csv('../results/ice_berg_svm.csv', index=False)