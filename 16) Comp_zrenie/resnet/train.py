import numpy as np
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_train(path):
    train_datagen = ImageDataGenerator(validation_split=0.25, rescale=1.0 / 255)
    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode="sparse",
        subset="training",
        seed=12345,
    )
    return train_datagen_flow


def create_model(input_shape):
    backbone = ResNet50(
        input_shape=(150, 150, 3),
        weights="/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
        include_top=False,
    )
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(12, activation="softmax"))
    optimizer = Adam()
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"]
    )
    return model


# Обучаем модель
def train_model(
    model,
    train_data,
    test_data,
    batch_size=None,
    epochs=10,
    steps_per_epoch=None,
    validation_steps=None,
):
    train_datagen_flow = train_data
    test_datagen_flow = test_data
    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
    )
    return model
