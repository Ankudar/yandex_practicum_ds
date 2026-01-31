from datetime import datetime

import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TARGET_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 16
RANDOM_SEED = 42
EARLY_STOP = 5
N_EPOCHS = 10
METRIC = "mae"
CLASS_MODE = "raw"


def load_train(path):
    labels_df = pd.read_csv(f"{path}/labels.csv")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.4,
    )

    return train_datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=f"{path}/final_files",
        x_col="file_name",
        y_col="real_age",
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset="training",
        seed=RANDOM_SEED,
        shuffle=True,
    )


def load_test(path):
    labels_df = pd.read_csv(f"{path}/labels.csv")

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.4,
    )

    return test_datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=f"{path}/final_files",
        x_col="file_name",
        y_col="real_age",
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset="validation",
        seed=RANDOM_SEED,
        shuffle=False,
    )


def create_model(input_shape, steps_per_epoch=None):
    backbone = ResNet50(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )

    backbone.trainable = True

    model = Sequential(
        [
            backbone,
            GlobalAveragePooling2D(),
            Dense(1, activation="relu"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
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
        steps_per_epoch = train_data.n // train_data.batch_size
    if validation_steps is None:
        validation_steps = test_data.n // test_data.batch_size

    checkpoint = ModelCheckpoint(
        "best_model.h5",
        monitor=f"val_{METRIC}",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    early_stopping = EarlyStopping(
        monitor=f"val_{METRIC}",
        patience=EARLY_STOP,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    )

    class DateTimeLogger(Callback):
        def on_epoch_begin(self, epoch, logs=None):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            print(
                f"\nEpoch {epoch + 1} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | LR={lr:.6f}"
            )

    model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
        callbacks=[
            checkpoint,
            early_stopping,
            DateTimeLogger(),
        ],
    )

    model.load_weights("best_model.h5")
    return model
