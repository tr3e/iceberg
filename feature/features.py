import sys
sys.path.append("..") 
import preprocess as pre 
from keras.models import Model, load_model
import numpy as np
from sklearn.model_selection import train_test_split


resnet = load_model('../net/model_best3.hdf5')

fex = Model(inputs=resnet.input, outputs=resnet.get_layer('flatten_1').output)


train_df = pre.load_data('train.json')
angles = pre.get_angles(train_df)[:, np.newaxis]
images = pre.get_images(train_df)
labels = pre.get_labels(train_df)
del(train_df)
img_gen = pre.images_generator()

e = 1
for x_batch, y_batch in img_gen.flow(images, labels, batch_size=10000, shuffle=False):
    print 'epoch:', e
    features = fex.predict(x_batch)
    features = np.concatenate((features, angles), axis=1).astype('float32')
    labels = y_batch
    np.save('features_train' + str(e) +'.npy', features)
    np.save('labels_train' + str(e) +'.npy', labels)
    e += 1

# features_train = fex.predict(images)
# features_train = np.concatenate((features_train, angles), axis=1).astype('float32')

# np.save('features_test.npy', features_test)
# np.save('features_train.npy', features_train)