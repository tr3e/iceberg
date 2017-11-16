import os
import numpy as np 
import pandas as pd 
import sys
sys.path.append("..") 
import preprocess as pre 
from resnext import build_resnext
from resnet import build_resnet
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from keras.optimizers import Nadam, Adamax, Adam, SGD, RMSprop
from sklearn import metrics
from sklearn.model_selection import train_test_split
from DPN import DPN1


train_df = pre.load_data('train.json')
images = pre.get_images(train_df)
labels = pre.get_labels(train_df)
del(train_df)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.1, random_state=0)

img_gen = pre.images_generator()
print 'Images generator inits completely'

print images.shape
print labels.shape

# clf = build_resnet((75, 75, 3), nb_classes=2, N=3, k=2, dropout=0.0, verbose=1)
clf = DPN1((75, 75, 3))
print 'Build successfully'
plot_model(clf, "resnet.png", show_shapes=True, show_layer_names=True)

# clf = load_model('model_best3.hdf5')


optimizer = Adamax(0.2)
clf.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print 'Compile successfully'


checkpoint = ModelCheckpoint('model_best4.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tb_cb = TensorBoard(log_dir='/tmp/keras_log', write_images=1, histogram_freq=1) 

callbacks_list = [checkpoint, tb_cb]

clf.fit_generator(img_gen.flow(X_train, y_train, batch_size=128), steps_per_epoch=10, epochs=10000, validation_data=(X_val, y_val), callbacks=callbacks_list)
#clf.fit(images, labels, epochs=100, batch_size=128, validation_split=0.2, callbacks=callbacks_list)
