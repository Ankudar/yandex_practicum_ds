from datetime import datetime

import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TARGET_SIZE = (150, 150)
INPUT_SHAPE = (150, 150, 3)
BATCH_SIZE = 16
RANDOM_SEED = 42
N_EPOCHS = 30
METRIC = "mae"
CLASS_MODE = "raw"


def load_train(path):
    labels_df = pd.read_csv(f"{path}/labels.csv")

    print(f"Загружен датафрейм с колонками: {labels_df.columns.tolist()}")
    print(f"Количество строк: {len(labels_df)}")

    train_datagen = ImageDataGenerator(validation_split=0.25, rescale=1.0 / 255)

    train_datagen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=f"{path}/final_files",
        x_col="file_name",
        y_col="real_age",
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset="training",
        seed=RANDOM_SEED,
    )
    return train_datagen_flow


def load_test(path):
    labels_df = pd.read_csv(f"{path}/labels.csv")

    test_datagen = ImageDataGenerator(validation_split=0.25, rescale=1.0 / 255)

    test_datagen_flow = test_datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=f"{path}/final_files",
        x_col="file_name",
        y_col="real_age",
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset="validation",
        seed=RANDOM_SEED,
    )
    return test_datagen_flow


def create_model(input_shape, steps_per_epoch=None):
    backbone = ResNet50(
        input_shape=INPUT_SHAPE,
        weights="/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
        include_top=False,
    )

    backbone.trainable = False

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation="linear"))

    initial_learning_rate = 0.01

    if steps_per_epoch is None:
        decay_steps = 100 * 2
    else:
        decay_steps = steps_per_epoch * 2  # Уменьшать каждые 2 эпохи

    lr_schedule = ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=0.9,  # Умножать на 0.9 каждый раз, т.е. уменьшаем LR
        staircase=True,
    )

    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[METRIC],
    )
    return model


def train_model(
    model,
    train_data,
    test_data,
    batch_size=None,
    epochs=N_EPOCHS,
    steps_per_epoch=None,
    validation_steps=None,
):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    # Сохранение лучшей модели
    checkpoint = ModelCheckpoint(
        "best_model.h5",
        monitor=f"val_{METRIC}",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    # Ранняя остановка если нет улучшений 5 эпох
    early_stopping = EarlyStopping(
        monitor=f"val_{METRIC}",
        patience=5,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    )

    class DateTimeLogger(Callback):
        def on_epoch_begin(self, epoch, logs=None):
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_lr = float(self.model.optimizer.learning_rate)
            print(f"\nНачало эпохи {epoch + 1} - {current_time}, LR: {current_lr:.6f}")

    date_time_logger = DateTimeLogger()

    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
        callbacks=[
            checkpoint,
            early_stopping,
            date_time_logger,
        ],
    )

    model.load_weights("best_model.h5")
    return model
