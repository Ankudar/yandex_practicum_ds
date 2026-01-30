import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.keras.layers import AvgPool2D, Conv2D, Dense, Flatten
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
    model = Sequential()
    model.add(
        Conv2D(6, (5, 5), padding="same", activation="relu", input_shape=input_shape)
    )
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=12, activation="softmax"))
    optimizer = Adam()
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"]
    )
    return model


def train_model(
    model,
    train_data,
    test_data,
    batch_size=None,
    epochs=1,
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


# if __name__ == "__main__":
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     DATA_PATH = os.path.join(BASE_DIR, "data")
#     os.makedirs(DATA_PATH, exist_ok=True)

#     if not os.path.exists(os.path.join(DATA_PATH, "train_features.npy")):
#         (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#         np.save(os.path.join(DATA_PATH, "train_features.npy"), x_train)
#         np.save(os.path.join(DATA_PATH, "train_target.npy"), y_train)
#         np.save(os.path.join(DATA_PATH, "test_features.npy"), x_test)
#         np.save(os.path.join(DATA_PATH, "test_target.npy"), y_test)

#     x_train, y_train = load_train(DATA_PATH + os.sep)

#     x_test = np.load(os.path.join(DATA_PATH, "test_features.npy"))
#     y_test = np.load(os.path.join(DATA_PATH, "test_target.npy"))
#     x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

#     model = create_model(input_shape=(28, 28, 1))
#     model = train_model(
#         model,
#         train_data=(x_train, y_train),
#         test_data=(x_test, y_test),
#         epochs=20,
#     )
