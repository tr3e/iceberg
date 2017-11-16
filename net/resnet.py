from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense, GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU


def initial_conv(input):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(input)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = LeakyReLU()(x)

    x = Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D()(x)

    return x


def expand_conv(init, base, k, strides=(1, 1)):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(init)
    x = Convolution2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = LeakyReLU()(x)

    x = Convolution2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = LeakyReLU()(x)

    skip = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(init)
    skip = Convolution2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal', use_bias=False)(skip)
    skip = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(skip)
    skip = LeakyReLU()(skip)

    m = Add()([x, skip])

    return m


def conv1_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Convolution2D(32* k, (3, 3), padding='same', kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = LeakyReLU()(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = LeakyReLU()(x)

    m = Add()([init, x])
    return m


def conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Convolution2D(64* k, (3, 3), padding='same', kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = LeakyReLU()(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = LeakyReLU()(x)

    m = Add()([init, x])
    return m


def conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Convolution2D(128* k, (3, 3), padding='same', kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = LeakyReLU()(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(128 * k, (3, 3), padding='same', kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = LeakyReLU()(x)

    m = Add()([init, x])
    return m


def conv4_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Convolution2D(256 * k, (3, 3), padding='same', kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = LeakyReLU()(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(256 * k, (3, 3), padding='same', kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = LeakyReLU()(x)

    m = Add()([init, x])
    return m


def build_resnet(input_dim, nb_classes=1, N=2, k=1, dropout=0.0, verbose=1):
    """
    Creates a Wide Residual Network with specified parameters
    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    ip = Input(shape=input_dim)
    x = GaussianNoise(0.01)(ip)
    x = initial_conv(ip)
    nb_conv = 4

    x = expand_conv(x, 32, k)

    for i in range(N - 1):
        x = conv1_block(x, k, dropout)
        nb_conv += 2

    x = expand_conv(x, 64, k, strides=(2, 2))

    x = GaussianNoise(0.01)(x)

    for i in range(N - 1):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    x = expand_conv(x, 128, k, strides=(2, 2))

    x = GaussianNoise(0.01)(x)

    for i in range(N - 1):
        x = conv3_block(x, k, dropout)
        nb_conv += 2

    x = expand_conv(x, 256, k, strides=(2, 2))

    x = GaussianNoise(0.01)(x)

    for i in range(N - 1):
        x = conv4_block(x, k, dropout)
        nb_conv += 2

    x = AveragePooling2D((5, 5))(x)
    x = Flatten()(x)

    # x = Dense(nb_classes, activation='softmax')(x)
    x = Dropout(0.5)(x)

    x = Dense(1, activation='sigmoid')(x)

    clf= Model(ip, x)

    if verbose: print("Residual Network-%d-%d created." % (nb_conv, k))
    return clf
