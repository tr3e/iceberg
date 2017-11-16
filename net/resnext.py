import numpy as np 
import pandas as pd 
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

channel_axis = 1 if K.image_data_format() == "channels_first" else -1

def conv_block(input, input_dim, nb_filters, k=32, strides=1, dropout=0.0):
    init = input
    paths = []

    for i in xrange(k):

        x = Convolution2D(nb_filters, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False)(input)
        x= BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        if dropout > 0.0: 
            x = Dropout(dropout)(x)

        x = Convolution2D(nb_filters, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x= BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        if dropout > 0.0: 
            x = Dropout(dropout)(x)

        x = Convolution2D(input_dim, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x= BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        paths.append(x)

    m = Add()(paths)
    m = Activation('relu')(m)
    m = Add()([init, m])
    return m


def initial_conv(input):
    x = Convolution2D(64, (7, 7), strides=3, padding='same', kernel_initializer='he_normal', use_bias=False)(input)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    return x


def expand_conv(input, input_dim, nb_filters, k=32, strides=2, dropout=0.0):
    init = input
    paths = []

    for i in xrange(k):

        x = Convolution2D(nb_filters, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(input)
        x= BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        if dropout > 0.0: 
            x = Dropout(dropout)(x)

        x = Convolution2D(nb_filters, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x= BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        if dropout > 0.0: 
            x = Dropout(dropout)(x)

        x = Convolution2D(input_dim * 2, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x= BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        paths.append(x)

    x = Convolution2D(input_dim * 2, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal', use_bias=False)(init)
    x= BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)

    m = Add()(paths)
    m = Activation('relu')(m)
    m = Add()([x, m])
    return m


def build_resnext(input_dim, nb_classes=1, dropout=0.3, verbose=1):
    input_layer = Input(shape=input_dim)
    x = initial_conv(input_layer)

    for i in range(2):
        x = conv_block(x, 64, 2, dropout=dropout)

    x = expand_conv(x, 64, 2, dropout=dropout)

    for i in range(2):
        x = conv_block(x, 128, 4, dropout=dropout)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((8, 8))(x)

    features = Flatten()(x)
    x = Dense(nb_classes, activation='sigmoid')(features)

    clf = Model(input_layer, x)
    fex = Model(input_layer, features)
    return clf, fex

if __name__ == "__main__":
    from keras.utils import plot_model
    from keras.layers import Input
    from keras.models import Model

    init = (75, 75, 2)

    wrn_28_10, features = build_resnext(init, nb_classes=2, dropout=0.0)
    print features

    #wrn_28_10.summary()

    #plot_model(wrn_28_10, "WRN-16-2.png", show_shapes=True, show_layer_names=True)
