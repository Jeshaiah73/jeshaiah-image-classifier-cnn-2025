# model.py
# Residual-inspired CNN with BatchNorm + Dropout
# Author: Jeshaiah Jesse

import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block(x, filters, kernel_size=3, strides=1):
    """Basic Conv2D -> BatchNorm -> ReLU block"""
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      strides=strides, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def residual_block(x, filters):
    """Simple residual block (2 conv layers + skip connection)"""
    shortcut = x
    x = conv_block(x, filters)
    x = conv_block(x, filters)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x


def build_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = conv_block(inputs, 64)
    x = conv_block(x, 64)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = conv_block(x, 128)
    x = residual_block(x, 128)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Block 3
    x = conv_block(x, 256)
    x = residual_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # Fully connected
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs,
                         name='jeshaiah_cifar10_cnn')
    return model
