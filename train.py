# train.py
"""
Training script for CIFAR-10 CNN.
Unique project id: jeshaiah-image-classifier-cnn-2025
Author: Jeshaiah Jesse
"""
import os
import argparse
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from model import build_model
from utils import get_cifar10_data, PlotHistory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--img-shape', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = get_cifar10_data()

    model = build_model(input_shape=x_train.shape[1:], num_classes=10)
    model.summary()

    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    ckpt_cb = callbacks.ModelCheckpoint(
        filepath=os.path.join(args.save_dir, 'best_model.h5'),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
    early_cb = callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)

    # Data augmentation pipeline (real-time)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.125,
        height_shift_range=0.125,
        zoom_range=0.1,
        rotation_range=15,
        fill_mode='reflect'
    )
    datagen.fit(x_train)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) // args.batch_size,
        callbacks=[ckpt_cb, early_cb, reduce_lr]
    )

    # Save final model
    model.save(os.path.join(args.save_dir, 'final_model.h5'))

    # Plot training curves
    PlotHistory(history.history).save('training_history.png')


if __name__ == '__main__':
    main()
