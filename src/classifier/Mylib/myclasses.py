from datetime import datetime, timedelta
from zipfile import ZipFile
import shutil
import urllib.request
import pickle
import numbers
import itertools
import re
import math
from box.exceptions import BoxValueError
import yaml
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import pickle
import plotly.express as px
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers


class ConvNetBlock_XceptionVersion(layers.Layer):
    """Gồm các layers sau:
    - SeparableConv2D
    - SeparableConv2D
    - MaxPooling2D

    Đi kèm là sự ết hợp giữa residual connections, batch normalization và  depthwise separable convolutions (lớp SeparableConv2D)

    Attributes:
        filters (_type_): số lượng filters trong lớp SeparableConv2D
    """

    def __init__(self, filters):
        """_summary_"""
        super(ConvNetBlock_XceptionVersion, self).__init__()
        self.filters = filters

        print("===========Khởi tạo thành công ============")

    def call(self, x):
        residual = x

        # First part of the block
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(self.filters, 3, padding="same", use_bias=False)(x)

        # Second part of the block
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(self.filters, 3, padding="same", use_bias=False)(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Apply residual connection
        residual = layers.Conv2D(
            self.filters, 1, strides=2, padding="same", use_bias=False
        )(residual)
        x = layers.add([x, residual])

        return x


class ConvNetBlock_Advanced(layers.Layer):
    """Gồm các layers sau:
    - Conv2D
    - Conv2D
    - MaxPooling2D

    Đi kèm là sự ết hợp giữa residual connections, batch normalization

    Attributes:
        filters (_type_): số lượng filters trong lớp Conv2D
        pooling (bool, optional): Có lớp MaxPooling2D không. Defaults to True.
    """

    def __init__(self, filters, pooling=True):
        super(ConvNetBlock_Advanced, self).__init__()
        self.filters = filters
        self.pooling = pooling

        print("===========Khởi tạo thành công ============")

    def call(self, x):
        residual = x

        # First part of the block
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(self.filters, 3, padding="same")(x)

        # Second part of the block
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(self.filters, 3, padding="same")(x)

        # Có dùng MaxPooling2D không
        if self.pooling:
            x = layers.MaxPooling2D(2, padding="same")(x)
            residual = layers.Conv2D(self.filters, 1, strides=2)(
                residual
            )  # Có dùng max pooling thì kèm cái này thôi
        elif self.filters != residual.shape[-1]:
            residual = layers.Conv2D(self.filters, 1)(
                residual
            )  # Nếu không dùng max pooling thì kèm cái này

        # Apply residual connection
        x = layers.add([x, residual])

        return x


class ConvNetBlock(layers.Layer):
    """Kết hợp các layers sau:
    - Conv2D * num_Conv2D_layer
    - MaxPooling

    Attributes:
        filters (_type_): số lượng filters trong lớp Conv2D
        num_Conv2D (int, optional): số lượng lớp num_Conv2D. Defaults to 1.
    """

    def __init__(self, filters, num_Conv2D=1):
        """ """
        super(ConvNetBlock, self).__init__()
        self.filters = filters
        self.num_Conv2D = num_Conv2D

    def call(self, x):
        for _ in range(self.num_Conv2D):
            x = layers.Conv2D(self.filters, 3, activation="relu")(x)

        x = layers.MaxPooling2D(pool_size=2)(x)

        return x


class DataPositionAugmentation(layers.Layer):
    """Tăng cường dữ liệu ở khía cạnh vị trí, bao gồm các lớp sau (**trong tf.keras.layers**)
    - RandomFlip
    - RandomRotation
    - RandomZoom

    Attributes:
    """

    def __init__(self, filters, num_Conv2D=1):
        """ """
        super(ConvNetBlock, self).__init__()
        self.filters = filters
        self.num_Conv2D = num_Conv2D

    def call(self, x):
        for _ in range(self.num_Conv2D):
            x = layers.Conv2D(self.filters, 3, activation="relu")(x)

        x = layers.MaxPooling2D(pool_size=2)(x)

        return x
