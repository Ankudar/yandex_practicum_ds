import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.keras.layers import AvgPool2D, Conv2D, Dense, Flatten

# Важно: используем только tensorflow.keras для всех импортов
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def load_train(path):
    features_train = np.load(path + "train_features.npy")
    target_train = np.load(path + "train_target.npy")
    # Важно: преобразовать в 4D формат для сверточной сети
    features_train = features_train.reshape(-1, 28, 28, 1) / 255.0
    return features_train, target_train


def create_model(input_shape):
    # Используем Adam с learning_rate (не lr)
    optimizer = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(
        Conv2D(6, (5, 5), padding="same", activation="relu", input_shape=input_shape)
    )
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding="valid", activation="relu"))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation="relu"))
    model.add(Dense(84, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model,
    train_data,
    test_data,
    batch_size=32,
    epochs=20,
):
    features_train, target_train = train_data
    features_test, target_test = test_data

    model.fit(
        features_train,
        target_train,
        validation_data=(features_test, target_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        shuffle=True,
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
