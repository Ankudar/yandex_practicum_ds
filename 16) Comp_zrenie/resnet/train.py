# import numpy as np
# from tensorflow.keras.applications.resnet import ResNet50
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.layers import (
#     BatchNormalization,
#     Dense,
#     Dropout,
#     GlobalAveragePooling2D,
# )
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


# def load_train(path):
#     train_datagen = ImageDataGenerator(
#         validation_split=0.25,
#         rescale=1 / 255.0,
#         horizontal_flip=True,
#         rotation_range=20,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         zoom_range=0.1,
#         shear_range=0.1,
#         brightness_range=[0.9, 1.1],
#         fill_mode="nearest",
#     )
#     train_datagen_flow = train_datagen.flow_from_directory(
#         path,
#         target_size=(150, 150),
#         batch_size=16,
#         class_mode="sparse",
#         subset="training",
#         seed=12345,
#     )
#     return train_datagen_flow


# def create_model(input_shape):
#     backbone = ResNet50(
#         input_shape=input_shape,
#         weights="/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
#         include_top=False,
#     )

#     backbone.trainable = False

#     model = Sequential(
#         [
#             backbone,
#             GlobalAveragePooling2D(),
#             Dense(256, activation="relu"),
#             BatchNormalization(),
#             Dropout(0.5),
#             Dense(128, activation="relu"),
#             BatchNormalization(),
#             Dropout(0.3),
#             Dense(12, activation="softmax"),
#         ]
#     )

#     optimizer = Adam(learning_rate=0.0001)

#     model.compile(
#         optimizer=optimizer,
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     print(f"Total layers: {len(model.layers)}")
#     trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
#     non_trainable_params = np.sum(
#         [np.prod(v.get_shape()) for v in model.non_trainable_weights]
#     )
#     print(f"Trainable parameters: {trainable_params:,}")
#     print(f"Non-trainable parameters: {non_trainable_params:,}")
#     print(f"Total parameters: {trainable_params + non_trainable_params:,}")

#     return model


# def train_model(
#     model,
#     train_data,
#     test_data,
#     batch_size=None,
#     epochs=3
# ):

#     # Улучшенные callbacks
#     callbacks = [
#         EarlyStopping(
#             monitor="val_accuracy",
#             patience=8,
#             restore_best_weights=True,
#             mode="max",
#             verbose=2,
#         ),
#         ReduceLROnPlateau(
#             monitor="val_loss",
#             factor=0.5,
#             patience=4,
#             min_lr=1e-7,
#             verbose=2,
#         ),
#     ]

#     history = model.fit(
#         train_data,
#         validation_data=test_data,
#         epochs=epochs,
#         callbacks=callbacks,
#         verbose=2,
#     )

#     return model


from datetime import datetime

import numpy as np
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
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
        seed=42,
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


def train_model(
    model,
    train_data,
    test_data,
    batch_size=None,
    epochs=10,
    steps_per_epoch=None,
    validation_steps=None,
):
    checkpoint = ModelCheckpoint(
        "best_model.h5",
        monitor="val_acc",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    class DateTimeLogger(Callback):
        def on_epoch_begin(self, epoch, logs=None):
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nНачало эпохи {epoch + 1} - {current_time}")

    date_time_logger = DateTimeLogger()

    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
        callbacks=[checkpoint, date_time_logger],
    )

    model.load_weights("best_model.h5")

    return model
