import numpy as np 
import pandas as pd 
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator

data_folder = '/workspace/lianghan/iceberg/ice_data/'


def get_images(data_frame):
    band1 = []
    for item in data_frame.band_1:
        band1.append(np.array(item).reshape([75, 75]))

    band2 = []
    for item in data_frame.band_2:
        band2.append(np.array(item).reshape((75, 75)))

    band1 = np.array(band1)
    band2 = np.array(band2)
    band3 = np.mean((band1, band2), axis=0)
    band1 = band1[:, :, :, np.newaxis]
    band2 = band2[:, :, :, np.newaxis]
    band3 = band3[:, :, :, np.newaxis]

    images = np.concatenate((band1, band2, band3), axis=3)
    new_images = []
    for image in images:
        new_images.append(cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC))
    return np.array(new_images)



def get_angles(data_frame):
    angles = np.array(data_frame['inc_angle'])
    if 'na' in angles:
        na_index = angles == 'na'
        angles_mean = angles[~na_index].mean()
        angles_std = angles[~na_index].std()
        angles[na_index] = 0.0
        angles[~na_index] = (angles[~na_index] - angles_mean) / angles_std
    else:
        angles_mean = angles.mean()
        angles_std = angles.std()
        angles = (angles - angles_mean) / angles_std
    return np.array(angles).astype('float32')


def get_labels(data_frame):
    labels = data_frame.is_iceberg
    return np.array(labels)


def load_data(filename):
    return pd.read_json(os.path.join(data_folder, filename))


def load_features(filename):
    features = np.load(filename).astype('float32')
    return features


def images_generator():
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=0.9,
        fill_mode='nearest')
    return datagen
