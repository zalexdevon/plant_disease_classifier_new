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
from classifier.Mylib.myclasses import CustomisedModelCheckpoint
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data_to_train(self):
        self.train_ds = tf.data.Dataset.load(self.config.train_ds_path)
        self.val_ds = tf.data.Dataset.load(self.config.val_ds_path)
        self.monitor = f"val_{self.config.scoring}"

        # Toàn bộ chỉ số đánh giá
        self.all_metrics = ["loss"] + self.config.metrics

        # List các model (layers)
        self.list_layers = [
            myfuncs.convert_list_string_to_list_object_20(layers)
            for layers in self.config.list_layers
        ]
        self.num_models = len(self.list_layers)

        # Tạo 1 list các optimzer, (mỗi model là 1 optimizer riềng)
        self.list_optimizer = [
            myfuncs.convert_string_to_object_4(self.config.optimizer)
            for _ in range(self.num_models)
        ]

        # Callbacks
        self.added_callbacks = [
            myfuncs.convert_string_to_object_4(callback)
            for callback in self.config.callbacks
        ]

        self.class_names = myfuncs.load_python_object(self.config.class_names_path)

    def load_callbacks(self):
        self.list_callbacks = []
        for i in range(self.num_models):
            callbacks = [
                # TqdmCallback(verbose=2),
                CustomisedModelCheckpoint(
                    filepath=os.path.join(
                        self.config.best_models_in_training_dir, f"{i}.keras"
                    ),
                    monitor=self.monitor,
                    indicator=self.config.target_score,
                ),
                TensorBoard(
                    log_dir=self.config.root_logs_dir,
                    histogram_freq=1,
                    write_graph=True,  # Đảm bảo ghi đồ thị
                    write_images=True,
                ),
            ] + self.added_callbacks

            self.list_callbacks.append(callbacks)

    def load_1model(self, layers, optimizer):
        inputs = layers[0]
        x = inputs

        for layer in layers[1:]:
            x = layer(x)

        model = keras.Model(inputs=inputs, outputs=x)

        model.compile(
            optimizer=optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics,
        )

        return model

    def load_model(self):
        self.models = [
            self.load_1model(layers, optimizer)
            for layers, optimizer in zip(self.list_layers, self.list_optimizer)
        ]

    def train_tfDataset(self):
        """Train với kdl = **tf.Dataset**"""
        tf.config.run_functions_eagerly(True)  # Bật eager execution
        tf.data.experimental.enable_debug_mode()  # Bật chế độ eager cho tf.data
        self.list_num_epochs = []

        print(
            f"\n========TIEN HANH TRAIN {self.num_models} MODELS !!!!!!================\n"
        )

        for index, model, callbacks in zip(
            list(range(self.num_models)), self.models, self.list_callbacks
        ):
            print(f"\n====== Tiến hành train model no. {index} ==========\n")
            history = model.fit(
                self.train_ds,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                verbose=1,
                validation_data=self.val_ds,
                callbacks=callbacks,
            )

            # Lưu lại số epoch đã chạy
            self.list_num_epochs.append(len(history.history["loss"]))
            print(f"\n====== Kết thúc train model no. {index}==========\n")

        print(
            f"\n========KET THUC TRAIN {self.num_models} MODELS !!!!!!================\n"
        )

    def load_trained_models_and_val_scorings(self):
        self.val_scorings = []
        self.trained_models = []
        index_scoring = self.all_metrics.index(self.config.scoring)
        for i in range(self.num_models):
            model = load_model(
                os.path.join(self.config.best_models_in_training_dir, f"{i}.keras")
            )
            scoring = model.evaluate(self.val_ds, verbose=0)[index_scoring]

            self.trained_models.append(model)
            self.val_scorings.append(scoring)

    def save_best_model_results(self):
        # Lấy model tốt nhất và số epoch tương ứng
        self.load_trained_models_and_val_scorings()
        index_best_model = (
            np.argmin(self.val_scorings)
            if self.config.scoring in myfuncs.SCORINGS_PREFER_MININUM
            else np.argmax(self.val_scorings)
        )
        self.best_model = self.trained_models[index_best_model]
        num_epochs = self.list_num_epochs[index_best_model]

        # Tìm các chỉ số đánh giá
        result = list(self.best_model.evaluate(self.train_ds, verbose=0))
        val_evaluation_metrics = ["val_" + item for item in self.all_metrics]
        val_result = list(self.best_model.evaluate(self.val_ds, verbose=0))
        self.results_dict = dict(
            zip(self.all_metrics + val_evaluation_metrics, result + val_result)
        )

        self.train_scoring = self.results_dict.pop(self.config.scoring)
        self.val_scoring = self.results_dict.pop("val_" + self.config.scoring)

        # Tìm classfication report
        train_classification_report = myfuncs.get_classification_report_for_DLmodel_21(
            model=self.best_model,
            ds=self.train_ds,
            class_names=self.class_names,
            batch_size=self.config.batch_size,
        )

        val_classification_report = myfuncs.get_classification_report_for_DLmodel_21(
            model=self.best_model,
            ds=self.val_ds,
            class_names=self.class_names,
            batch_size=self.config.batch_size,
        )

        # In ra các kết quả đánh giá
        self.best_model_results_text = (
            "========KẾT QUẢ MÔ HÌNH TỐT NHẤT================\n"
        )
        self.best_model_results_text += "Chỉ số scoring\n"
        self.best_model_results_text += (
            f"Train {self.config.scoring}: {self.train_scoring}\n"
        )
        self.best_model_results_text += (
            f"Val {self.config.scoring}: {self.val_scoring}\n"
        )

        self.best_model_results_text += "\nCác chỉ số khác\n"
        for key, value in self.results_dict.items():
            self.best_model_results_text += f"- {key}: {value}\n"

        self.best_model_results_text += (
            f"\nNUM_RUNNING_EPOCHS: {num_epochs} / {self.config.epochs}\n\n"
        )

        self.best_model_results_text += "Train classfication report: \n"
        self.best_model_results_text += f"{train_classification_report}\n\n"
        self.best_model_results_text += "Val classfication report: \n"
        self.best_model_results_text += f"{val_classification_report}\n\n"

        print(self.best_model_results_text)

        # Ghi kết quả đánh giá vào file results.txt
        with open(self.config.results_path, mode="w") as file:
            file.write(self.best_model_results_text)

        # Lưu cấu trúc của model
        keras.utils.plot_model(
            self.best_model, self.config.model_structure_path, show_shapes=True
        )

    def save_list_monitor_components(self):
        if self.config.scoring == "accuracy":
            self.train_scoring, self.val_scoring = (
                self.train_scoring * 100,
                self.val_scoring * 100,
            )

        if os.path.exists(self.config.list_monitor_components_path):
            self.list_monitor_components = myfuncs.load_python_object(
                self.config.list_monitor_components_path
            )

        else:  # Này là lần đầu training rồi !!!
            self.list_monitor_components = []

        # Lấy kết quả của model để hiện lên tooltip
        self.list_monitor_components += [
            (
                self.config.model_name,
                self.train_scoring,
                self.val_scoring,
            )
        ]

        myfuncs.save_python_object(
            self.config.list_monitor_components_path, self.list_monitor_components
        )
