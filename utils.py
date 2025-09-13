# utils.py
# Utility functions for CIFAR-10 training
# Author: Jeshaiah Jesse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_cifar10_data():
    """Load CIFAR-10 dataset and normalize to [0,1]."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train.squeeze()), (x_test, y_test.squeeze())


class PlotHistory:
    """Plot and save training history (loss & accuracy)."""
    def __init__(self, history):
        self.history = history

    def save(self, filename='training_history.png'):
        plt.figure(figsize=(8, 4))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='train_loss')
        plt.plot(self.history['val_loss'], label='val_loss')
        plt.legend()
        plt.title('Loss')

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history['accuracy'], label='train_acc')
        plt.plot(self.history['val_accuracy'], label='val_acc')
        plt.legend()
        plt.title('Accuracy')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
