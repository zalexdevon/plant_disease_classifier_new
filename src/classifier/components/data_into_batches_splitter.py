from classifier.entity.config_entity import DataIntoBatchesSplitterConfig
import pandas as pd
import os
from classifier import logger
from classifier.Mylib import myfuncs
from sklearn import metrics
from keras.models import load_model
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report


class DataIntoBatchesSplitter:
    def __init__(self, config: DataIntoBatchesSplitterConfig):
        self.config = config

    def load_data_and_split(self):
        # Doc dataset
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.config.folder_path,
            shuffle=True,
            image_size=(self.config.image_size, self.config.image_size),
            batch_size=self.config.batch_size,
        )

        # Chia thanh 3 tap
        train_ds, val_ds, test_ds = myfuncs.split_tfdataset_into_tranvaltest_1(
            dataset, train_size=self.config.train_size, val_size=self.config.val_size
        )

        train_ds = myfuncs.cache_prefetch_tfdataset_2(train_ds)
        val_ds = myfuncs.cache_prefetch_tfdataset_2(val_ds)
        test_ds = myfuncs.cache_prefetch_tfdataset_2(test_ds)

        # Lưu dữ liệu
        train_ds.save(self.config.train_ds_path)
        val_ds.save(self.config.val_ds_path)
        test_ds.save(self.config.test_ds_path)
        myfuncs.save_python_object(self.config.class_names_path, dataset.class_names)
