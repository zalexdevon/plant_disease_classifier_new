import pandas as pd
import os
from classifier import logger
from classifier.entity.config_entity import ModelEvaluationConfig
from classifier.Mylib import myfuncs
from sklearn import metrics
from keras.models import load_model
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_data_to_test(self):
        self.test_ds = tf.data.Dataset.load(self.config.test_ds_path)
        self.class_names = myfuncs.load_python_object(self.config.class_names_path)
        self.model = load_model(self.config.model_path)

    def evaluate_model(self):
        # Tìm các chỉ số loss, metrics
        evaluation_metrics = ["loss"] + self.config.metrics
        result = list(self.model.evaluate(self.test_ds, verbose=0))
        self.results_dict = dict(zip(evaluation_metrics, result))

        # Tìm classfication report
        classification_report = self.get_classification_report_for_best_model(
            self.test_ds
        )

        # In ra các kết quả đánh giá
        model_results_text = (
            "========KET QUA CUA MO HINH TREN TAP TEST================\n"
        )
        model_results_text += "CAC CHI SO DANH GIA:\n"
        for key, value in self.results_dict.items():
            model_results_text += f"- {key}: {value}\n"

        model_results_text += "\nCLASSIFICATION REPORT\n"
        model_results_text += classification_report + "\n\n"

        print(model_results_text)

        # Ghi kết quả đánh giá vào file results.txt
        with open(self.config.results_path, mode="w") as file:
            file.write(model_results_text)

    def get_classification_report_for_best_model(self, ds):
        y_true = []
        y_pred = []

        class_names = np.asarray(self.class_names)

        # Lặp qua các batch trong ds
        for images, labels in ds:
            # Dự đoán bằng mô hình
            predictions = self.model.predict(
                images, batch_size=self.config.batch_size, verbose=0
            )

            # Thêm vào danh sách
            y_pred_batch = class_names[np.argmax(predictions, axis=-1)].tolist()
            y_true_batch = class_names[np.asarray(labels)].tolist()
            y_true += y_true_batch
            y_pred += y_pred_batch

        return classification_report(y_true, y_pred)
