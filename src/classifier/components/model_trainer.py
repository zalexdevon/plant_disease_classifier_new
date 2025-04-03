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
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data_to_train(self):
        self.train_ds = tf.data.Dataset.load(self.config.train_ds_path)
        self.val_ds = tf.data.Dataset.load(self.config.val_ds_path)
        self.monitor = f"val_{self.config.scoring}"

        # Callbacks
        self.callbacks = [
            myfuncs.convert_string_to_object_4(callback)
            for callback in self.config.callbacks
        ]

        # Layers
        self.layers = [
            myfuncs.convert_string_to_object_4(layer)
            for layer in self.config.list_layers[0]
        ]

        self.class_names = myfuncs.load_python_object(self.config.class_names_path)
        self.optimizer = myfuncs.convert_string_to_object_4(self.config.optimizer)

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
        ] + self.callbacks

    def load_model(self):
        inputs = self.layers[0]
        x = inputs

        for layer in self.layers[1:]:
            x = layer(x)

        self.model = keras.Model(inputs=inputs, outputs=x)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics,
        )

    def train_tfDataset(self):
        """Train với kdl = **tf.Dataset**"""
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

    def find_scoring_val_scoring_for_best_model(self):
        best_model_train_score = self.results_dict[self.config.scoring]
        best_model_val_score = self.results_dict["val_" + self.config.scoring]

        while True:
            if self.config.scoring == "accuracy":
                best_model_train_score = best_model_train_score * 100
                best_model_val_score = best_model_val_score * 100

                break

            break

        return best_model_train_score, best_model_val_score

    def save_best_model_results(self):
        """Lưu các kết quả của model **tốt nhất** (đã được lưu trước đó bởi callback **ModelCheckpoint**)"""
        # Lấy model tốt nhất
        self.best_model = load_model(self.config.best_model_path)

        # Tìm các chỉ số loss, metrics
        evaluation_metrics = ["loss"] + self.config.metrics
        result = list(self.model.evaluate(self.train_ds, verbose=0))
        val_evaluation_metrics = ["val_" + item for item in evaluation_metrics]
        val_result = list(self.model.evaluate(self.val_ds, verbose=0))
        self.results_dict = dict(
            zip(evaluation_metrics + val_evaluation_metrics, result + val_result)
        )

        # Tìm classfication report
        train_classification_report = self.get_classification_report_for_best_model(
            self.train_ds
        )

        val_classification_report = self.get_classification_report_for_best_model(
            self.val_ds
        )

        # Tìm số vòng lặp đã chạy trước khi dừng
        num_epochs = len(self.history["loss"])

        # In ra các kết quả đánh giá
        self.best_model_results_text = (
            "========KET QUA CUA MO HINH TOT NHAT================\n"
        )
        self.best_model_results_text += "CAC CHI SO DANH GIA:\n"

        for key, value in self.results_dict.items():
            self.best_model_results_text += f"- {key}: {value}\n"

        self.best_model_results_text += (
            f"\nNUM_RUNNING_EPOCHS: {num_epochs} / {self.config.epochs}\n\n"
        )

        self.best_model_results_text += "\nCLASSIFICATION REPORT\n"

        self.best_model_results_text += "Train: \n"
        self.best_model_results_text += train_classification_report + "\n\n"
        self.best_model_results_text += "Val: \n"
        self.best_model_results_text += val_classification_report + "\n\n"

        print(self.best_model_results_text)

        # Ghi kết quả đánh giá vào file results.txt
        with open(self.config.results_path, mode="w") as file:
            file.write(self.best_model_results_text)

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
        keras.utils.plot_model(
            self.model, self.config.model_structure_path, show_shapes=True
        )

    def get_classification_report_for_best_model(self, ds):
        y_true = []
        y_pred = []

        class_names = np.asarray(self.class_names)

        # Lặp qua các batch trong train_ds
        for images, labels in ds:
            # Dự đoán bằng mô hình
            predictions = self.best_model.predict(
                images, batch_size=self.config.batch_size, verbose=0
            )

            y_pred_batch = class_names[np.argmax(predictions, axis=-1)].tolist()
            y_true_batch = class_names[np.asarray(labels)].tolist()
            y_true += y_true_batch
            y_pred += y_pred_batch

        return classification_report(y_true, y_pred)

    def save_list_monitor_components(self):
        # Tìm scoring cho tập train, val -> để hiện lên biểu đồ
        self.best_model_train_score, self.best_model_val_score = (
            self.find_scoring_val_scoring_for_best_model(self.results_dict)
        )

        if os.path.exists(self.config.list_monitor_components_path):
            self.list_monitor_components = myfuncs.load_python_object(
                self.config.list_monitor_components_path
            )

        else:
            self.list_monitor_components = []

        # Lấy kết quả của model để hiện lên tooltip
        self.list_monitor_components += [
            (
                self.config.model_name,
                self.best_model_train_score,
                self.best_model_val_score,
                self.best_model_results_text,
            )
        ]

        myfuncs.save_python_object(
            self.config.list_monitor_components_path, self.list_monitor_components
        )
