import os
import numpy as np 
import pandas as pd 
import sys
sys.path.append("..") 
import preprocess as pre
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from keras.models import load_model, Model
from sklearn.externals import joblib
from sklearn.utils import shuffle

data_folder = '/home/itsuki/workspace/iceberg/data'

X_train = pre.load_features('../feature/features_train1.npy')
y_train = pre.load_features('../feature/labels_train1.npy')
shuffle(X_train, y_train, random_state=14)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=None)
for i in xrange(2, 100):
    featrues_file = '../feature/features_train' + str(i) + '.npy'
    labels_file = '../feature/labels_train' + str(i) + '.npy'

    if os.path.exists(featrues_file):
        new_X = pre.load_features(featrues_file)
        new_y = pre.load_features(labels_file)
        shuffle(new_X, new_y, random_state=14)

        new_X, new_X_val, new_y, new_y_val = train_test_split(new_X, new_y, test_size=0.2, random_state=None)

        X_train = np.concatenate((X_train, new_X), axis=0).astype('float32')
        y_train = np.concatenate((y_train, new_y), axis=0).astype('float32')

        X_val = np.concatenate((X_val, new_X_val), axis=0).astype('float32')
        y_val = np.concatenate((y_val, new_y_val), axis=0).astype('float32')
    else:
        print 'no such file:', featrues_file
        break

print X_train.shape, y_train.shape

clf = RandomForestClassifier(n_estimators=1000, max_depth=100, oob_score=True, verbose=10, n_jobs=4, random_state=None)

clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_val)[:, 1]
print clf.oob_score_
print metrics.log_loss(y_val, y_pred)

# test_df = pre.load_data('test.json')
# features_test = pre.load_features('../feature/features_test.npy')

# results = clf.predict_proba(features_test)[:, 1]
# print results

# submission=pd.DataFrame({'id':test_df['id'], 'is_iceberg':results})
# submission.to_csv('../results/ice_berg_xgboost.csv', index=False)