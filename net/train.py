import os
import numpy as np 
import pandas as pd 
import sys
sys.path.append("..") 
import preprocess as pre 
from resnext import build_resnext
from resnet import build_resnet
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback
from keras_contrib.utils.save_load_utils import save_all_weights, load_all_weights
from keras.utils import plot_model
from keras.optimizers import Nadam, Adamax, Adam, SGD, RMSprop
from sklearn import metrics
from sklearn.model_selection import train_test_split
import dpn
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'

train_df = pre.load_data('train.json')
images = pre.get_images(train_df)
labels = pre.get_labels(train_df)
del(train_df)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=None)

img_gen = pre.images_generator()
print 'Images generator inits completely'

print 'training images:', X_train.shape
print 'validation images:', X_val.shape
print 'training labels:', y_train.shape
print 'validaiton labels:', y_val.shape

# clf = build_resnet((75, 75, 3), nb_classes=2, N=3, k=2, dropout=0.0, verbose=1)
clf = dpn.DPN137((224, 224, 3), classes=1)
clf.summary()
time.sleep(10000)
clf.load_weights('model-15-0.72.h5')
print 'Build successfully'
# plot_model(clf, "dpn-137.png", show_shapes=False, show_layer_names=True)

# clf = load_model('model_best3.hdf5')


optimizer = SGD(lr=0.00001, decay=0.5)
clf.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print 'Compile successfully'


checkpoint = ModelCheckpoint('model-{epoch:02d}-{val_loss:.4f}.h5', save_best_only=True, verbose=1, period=5, save_weights_only=True)
# tb_cb = TensorBoard(log_dir='/tmp/keras_log', write_images=0, histogram_freq=0) 

callbacks_list = [checkpoint]

clf.fit_generator(img_gen.flow(X_train, y_train, batch_size=16), steps_per_epoch=80, epochs=10000, validation_data=(X_val, y_val), callbacks=callbacks_list)
#clf.fit(images, labels, epochs=100, batch_size=128, validation_split=0.2, callbacks=callbacks_list)
