from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV 
from sklearn.pipeline import Pipeline 
from sklearn.utils import shuffle
from sklearn import metrics
import sys
sys.path.append("..") 
import preprocess as pre
import xgboost as xgb
import numpy as np
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
data_folder = '/home/itsuki/workspace/iceberg/data'

X = pre.load_features('../feature/features_train.npy')
y = pre.load_features('../feature/labels_train.npy')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=None)
# for i in xrange(2, 100):
#     featrues_file = '../feature/features_train' + str(i) + '.npy'
#     labels_file = '../feature/labels_train' + str(i) + '.npy'

#     if os.path.exists(featrues_file):
#         new_X = pre.load_features(featrues_file)
#         new_y = pre.load_features(labels_file)
#         shuffle(new_X, new_y, random_state=14)
#         new_X, new_X_val, new_y, new_y_val = train_test_split(new_X, new_y, test_size=0.2, random_state=1314)

#         X_train = np.concatenate((X_train, new_X), axis=0).astype('float32')
#         y_train = np.concatenate((y_train, new_y), axis=0).astype('float32')

#         X_val = np.concatenate((X_val, new_X_val), axis=0).astype('float32')
#         y_val = np.concatenate((y_val, new_y_val), axis=0).astype('float32')
#     else:
#         print 'no such file:', featrues_file
#         break

print X_train.shape, y_train.shape

clf = xgb.XGBClassifier(max_depth=100, 
                        learning_rate=0.05, 
                        n_estimators=1000, 
                        silent=True, 
                        objective='binary:logistic', 
                        #booster='gbtree',
                        #n_jobs=1, 
                        nthread=4, 
                        gamma=0, 
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=1, 
                        colsample_bytree=1, 
                        colsample_bylevel=1, 
                        reg_alpha= 3.8, 
                        reg_lambda=0, 
                        scale_pos_weight=1, 
                        base_score=0.5, 
                        seed=0, 
                        missing=None)


# grid_search = GridSearchCV(estimator=clf, param_grid=parameters, n_jobs = 4, verbose=100000, scoring='neg_log_loss', cv=5)

# grid_search.fit(features, labels)
# print grid_scores_, grid_search.best_score_, grid_search.best_score_

clf.fit(X_train, y_train, eval_metric=['logloss'], early_stopping_rounds=20, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True,)
print metrics.log_loss(y_train, clf.predict_proba(X_train)), metrics.log_loss(y_val, clf.predict_proba(X_val))

# clf.fit(features, labels, eval_metric=['logloss'], early_stopping_rounds=20, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True,)
# test_df = pre.load_data('test.json')

# features_test = pre.load_features('../feature/features_test.npy')

# results = clf.predict_proba(features_test)[:, 1]
# print results

# submission=pd.DataFrame({'id':test_df['id'], 'is_iceberg':results})
# submission.to_csv('../results/ice_berg_xgboost.csv', index=False)
