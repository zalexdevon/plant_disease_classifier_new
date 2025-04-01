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
from tensorflow.keras.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
)
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras.models import load_model


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data_to_train(self):
        self.train_ds = tf.data.Dataset.load(self.config.train_ds_path)
        self.val_ds = tf.data.Dataset.load(self.config.val_ds_path)
        self.monitor = f"val_{self.config.scoring}"

    def load_callbacks(self):
        self.callbacks = [
            # TqdmCallback(verbose=2),
            ModelCheckpoint(
                filepath=self.config.best_model_path,
                monitor=self.monitor,
                save_best_only=True,
            ),
            TensorBoard(
                log_dir=self.config.root_logs_dir,
                histogram_freq=1,
                write_graph=True,  # Đảm bảo ghi đồ thị
                write_images=True,
            ),
        ] + self.config.callbacks

    def load_model(self):
        inputs = keras.Input(shape=(self.config.image_size, self.config.image_size, 3))

        resize_layer = keras.layers.Resizing(
            self.config.image_size, self.config.image_size
        )(inputs)

        first_layer = self.config.layers[0]
        middle_layers = self.config.layers[1:]
        last_layer = Dense(units=len(self.config.class_names), activation="softmax")

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
        print("========TIEN HANH TRAIN MODEL !!!!!!================")

        self.history = self.model.fit(
            self.train_ds,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=1,
            validation_data=self.val_ds,
            callbacks=self.callbacks,
        ).history

        print("========KET THUC TRAIN MODEL !!!!!!================")

    def find_index_for_best_model(self):
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

        return best_model_index

    def find_scoring_val_scoring_for_best_model(self, results):
        best_model_train_score = results[self.config.scoring]
        best_model_val_score = results["val_" + self.config.scoring]

        while True:
            if self.config.scoring == "accuracy":
                best_model_train_score = best_model_train_score * 100
                best_model_val_score = best_model_val_score * 100

                break

            break

        return best_model_train_score, best_model_val_score

    def save_model(self):
        best_model_index = self.find_index_for_best_model()

        # Tìm các chỉ số đánh giá tương ứng
        results = {
            "loss": self.history["loss"][best_model_index],
            "val_loss": self.history["val_loss"][best_model_index],
        }

        for metric in self.config.metrics:
            results[metric] = self.history[metric][best_model_index]
            results[f"val_{metric}"] = self.history[f"val_{metric}"][best_model_index]

        num_epochs = len(self.history["loss"])

        self.best_model_train_score, self.best_model_val_score = (
            self.find_scoring_val_scoring_for_best_model(results)
        )

        # In ra các kết quả đánh giá
        best_model_results = "========KET QUA CUA MO HINH TOT NHAT================\n"
        best_model_results = "CAC CHI SO DANH GIA:\n"

        for key, value in results.items():
            best_model_results += f"- {key}: {value}\n"

        best_model_results += (
            f"\nNUM_RUNNING_EPOCHS: {num_epochs} / {self.config.epochs}\n\n"
        )

        # TODO: d
        print("Buocwcs vào CLASSIFICATION REPORT")
        # d

        try:
            best_model_results += "\nCLASSIFICATION REPORT\n"
            train_classification_report = self.get_classification_report_for_best_model(
                self.train_ds
            )
            val_classification_report = self.get_classification_report_for_best_model(
                self.val_ds
            )

            # TODO: d
            print("train_classification_report: \n" + train_classification_report)
            print()
            print("val_classification_report: \n" + val_classification_report)
            # d

            best_model_results += "Train: \n"
            best_model_results += train_classification_report + "\n\n"
            best_model_results += "Val: \n"
            best_model_results += val_classification_report + "\n\n"

            print(best_model_results)
        except Exception as e:
            print(f"Lỗi : {e}")

        print("CLASSIFICATION REPORT thành công !!!!!!")

        # Ghi kết quả đánh giá vào file results.txt
        with open(self.config.results_path, mode="w") as file:
            file.write(best_model_results)

        # Lưu các biểu đồ per epoch cho từng chỉ số (mỗi epoch là 1 model riêng)
        epochs = range(1, num_epochs + 1)
        epochs = [str(i) for i in epochs]

        metrics_and_loss = self.config.metrics + ["loss"]

        for item in metrics_and_loss:
            plt.plot(epochs, self.history[item], color="gray")
            plt.plot(epochs, self.history["val_" + item], color="blue")
            plt.ylim(bottom=0)
            plt.savefig(
                os.path.join(self.config.root_dir, item + "_per_epoch.png"),
                dpi=None,
                bbox_inches="tight",
                format=None,
            )
            plt.clf()

        # Lưu cấu trúc của model
        keras.utils.plot_model(self.model, self.config.structure_path, show_shapes=True)

    def get_classification_report_for_best_model(self, ds):
        y_true = []
        y_pred = []

        class_names = np.asarray(self.config.class_names)

        # Lấy model tốt nhất
        self.best_model = load_model(self.config.best_model_path)

        # Lặp qua các batch trong train_ds
        for images, y_true_batch in ds:
            # Dự đoán bằng mô hình
            predictions = self.best_model.predict(
                images, batch_size=self.config.batch_size
            )

            y_pred_batch = class_names[np.argmax(predictions, axis=-1)]

            # Thêm vào danh sách
            y_true.append(y_true_batch)
            y_pred.append(y_pred_batch)

        # In ra báo cáo phân loại
        return classification_report(y_true, y_pred)

    def save_list_monitor_components(self):
        if self.config.is_first_time == "f":

            self.list_monitor_components = myfuncs.load_python_object(
                self.config.list_monitor_components_path
            )

        else:
            self.list_monitor_components = []

        self.list_monitor_components += [
            (
                self.config.model_name,
                self.best_model_train_score,
                self.best_model_val_score,
            )
        ]

        myfuncs.save_python_object(
            self.config.list_monitor_components_path, self.list_monitor_components
        )
