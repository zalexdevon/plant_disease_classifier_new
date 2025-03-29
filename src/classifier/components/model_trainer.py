import pandas as pd
import os
from classifier.entity.config_entity import ModelTrainerConfig
from classifier.Mylib import myfuncs
import numpy as np
import random
import time
from tqdm.keras import TqdmCallback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data_to_train(self):
        self.train_ds = tf.data.Dataset.load(self.config.train_ds_path)
        self.val_ds = tf.data.Dataset.load(self.config.val_ds_path)
        self.monitor = f"val_{self.config.scoring}"

    def load_callbacks(self):
        self.callbacks = [
            TqdmCallback(verbose=1),
            ModelCheckpoint(
                filepath=self.config.best_model_path,
                monitor=self.monitor,
                save_best_only=True,
            ),
        ] + self.config.callbacks

    def load_model(self):
        inputs = keras.Input(shape=(self.config.image_size, self.config.image_size, 3))

        resize_layer = keras.layers.Resizing(
            self.config.image_size, self.config.image_size
        )(inputs)

        first_layer = self.config.layers[0]
        middle_layers = self.config.layers[1:-1]
        last_layer = self.config.layers[-1]

        x = first_layer(resize_layer)

        for layer in middle_layers:
            x = layer(x)

        outputs = last_layer(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=self.config.optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics,
        )

    def train(self):
        self.history = self.model.fit(
            self.train_ds,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=0,  # Tắt verbose của keras thay bằng tqdm
            validation_data=self.val_ds,
            callbacks=self.callbacks,
        ).history

    def save_model(self):
        # Tìm index ứng với best model
        best_model_index = None
        while True:
            if self.config.scoring == "accuracy":
                best_model_index = np.argmax(self.history[self.monitor])
                break
            if self.config.scoring == "loss":
                best_model_index = np.argmin(self.history[self.monitor])
                break

            break

        # Tìm các chỉ số đánh giá tương ứng
        results = {
            "loss": self.history["loss"][best_model_index],
            "val_loss": self.history["val_loss"][best_model_index],
        }

        for metric in self.config.metrics:
            results[metric] = self.history[metric][best_model_index]
            results[f"val_{metric}"] = self.history[f"val_{metric}"][best_model_index]

        num_epochs = len(self.history["loss"])

        # In ra các kết quả đánh giá
        print("========KET QUA MO HINH TOT NHAT================")
        for key, value in results.items():
            print(f"{key}: {value}")
        print(f"Số epochs đã chạy: {num_epochs} / {self.config.epochs}")

        # Ghi vào file
        content = "SCORING\n"

        for key, value in results.items():
            content += f"- {key}: {value}\n"

        content += f"\nNUM_EPOCHS: {num_epochs} / {self.config.epochs}\n\n"

        with open(self.config.results_path, mode="w") as file:
            file.write(content)

        # Lưu cấu trúc của model
        keras.utils.plot_model(self.model, self.config.structure_path, show_shapes=True)

    def save_list_monitor_components(self):

        if self.config.is_first_time == "f":

            self.list_monitor_components = myfuncs.load_python_object(
                self.config.list_monitor_components_path
            )

        else:
            self.list_monitor_components = []

        self.list_monitor_components += [
            (
                self.monitor_desc,
                self.train_score_follow_best_val,
                self.best_val_score,
            )
        ]

        myfuncs.save_python_object(
            self.config.list_monitor_components_path, self.list_monitor_components
        )
