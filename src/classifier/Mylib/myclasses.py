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
from tensorflow import keras
import keras_cv


class ConvNetBlock_XceptionVersion(layers.Layer):
    """Gồm các layers sau:
    - SeparableConv2D
    - SeparableConv2D
    - MaxPooling2D

    Đi kèm là sự kết hợp giữa residual connections, batch normalization và  depthwise separable convolutions (lớp SeparableConv2D)

    Attributes:
        filters (_type_): số lượng filters trong lớp SeparableConv2D
    """

    def __init__(self, filters):
        """_summary_"""
        # super(ConvNetBlock_XceptionVersion, self).__init__()
        super().__init__()
        self.filters = filters

    def build(self, input_shape):
        print(
            "============== Build class ConvNetBlock_XceptionVersion ================="
        )
        super().build(input_shape)

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

        print(
            "============= Đã qua được lớp ConvNetBlock_XceptionVersion ==============="
        )

        return x


class ConvNetBlock_Advanced(layers.Layer):
    """Gồm các layers sau:
    - Conv2D
    - Conv2D
    - MaxPooling2D

    Đi kèm là sự kết hợp giữa residual connections, batch normalization

    Attributes:
        filters (_type_): số lượng filters trong lớp Conv2D
        pooling (bool, optional): Có lớp MaxPooling2D không. Defaults to True.
    """

    def __init__(self, filters, pooling=True):
        # super(ConvNetBlock_Advanced, self).__init__()
        super().__init__()
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

        print("============= Đã qua được lớp ConvNetBlock_Advanced ===============")

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
        # super(ConvNetBlock, self).__init__()
        super().__init__()
        self.filters = filters
        self.num_Conv2D = num_Conv2D

    def call(self, x):
        for _ in range(self.num_Conv2D):
            x = layers.Conv2D(self.filters, 3, activation="relu")(x)

        x = layers.MaxPooling2D(pool_size=2)(x)

        print("============= Đã qua được lớp ConvNetBlock ===============")

        return x


class ImageDataPositionAugmentation(layers.Layer):
    """Tăng cường dữ liệu hình ảnh ở khía cạnh vị trí, bao gồm các lớp sau (**trong tf.keras.layers**)
    - RandomFlip
    - RandomRotation
    - RandomZoom

    Attributes:
        rotation_factor (float): Tham số cho lớp RandomRotation. Default to 0.2
        zoom_factor (float): Tham số cho lớp RandomZoom. Default to 0.2
    """

    def __init__(self, rotation_factor=0.2, zoom_factor=0.2):
        # super(ImageDataPositionAugmentation, self).__init__()
        super().__init__()
        self.rotation_factor = rotation_factor
        self.zoom_factor = zoom_factor

    def call(self, x):
        x = layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = layers.RandomRotation(factor=self.rotation_factor)(x)
        x = layers.RandomZoom(height_factor=self.zoom_factor)(x)

        print(
            "============= Đã qua được lớp ImageDataPositionAugmentation ==============="
        )

        return x


class ImageDataColorAugmentation(layers.Layer):
    """Tăng cường dữ liệu hình ảnh ở khía cạnh màu sắc, bao gồm các lớp sau (**trong keras_cv.layers**)
    - RandomBrightness
    - RandomGaussianBlur
    - RandomContrast
    - RandomHue
    - RandomSaturation

    Attributes:
        brightness_factor (float, optional): factor cho RandomBrightness. Defaults to 0.2.
        contrast_factor (float, optional): factor cho RandomContrast. Defaults to 0.2.
        hue_factor (float, optional): factor cho RandomHue. Defaults to 0.2.
        saturation_factor (float, optional): factor cho RandomSaturation. Defaults to 0.2.
    """

    def __init__(
        self,
        brightness_factor=0.2,
        contrast_factor=0.2,
        hue_factor=0.2,
        saturation_factor=0.2,
    ):
        # super(ImageDataColorAugmentation, self).__init__()
        super().__init__()
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.hue_factor = hue_factor
        self.saturation_factor = saturation_factor

    def call(self, x):
        x = keras_cv.layers.RandomBrightness(factor=self.brightness_factor)(x)
        x = keras_cv.layers.RandomGaussianBlur(kernel_size=3, factor=(0.0, 1.0))(
            x
        )  # Lớp này để mặc định
        x = keras_cv.layers.RandomContrast(
            factor=self.contrast_factor,
            value_range=(1 - self.contrast_factor, 1 + self.contrast_factor),
        )(x)
        x = keras_cv.layers.RandomHue(
            factor=self.hue_factor,
            value_range=(1 - self.hue_factor, 1 + self.hue_factor),
        )(x)
        x = keras_cv.layers.RandomSaturation(factor=self.saturation_factor)(x)

        print(
            "============= Đã qua được lớp ImageDataColorAugmentation ==============="
        )

        return x


class PretrainedModel(layers.Layer):
    """Sử dụng các pretrained models ở trong **keras.applications**
    Attributes:
        name (str): Tên model
        num_trainable (int, optional): Số lượng các lớp đầu tiên cho trainable = True. Defaults to 0.
    """

    def __init__(
        self,
        name,
        num_trainable=0,
    ):
        if num_trainable < 0:
            raise ValueError(
                "=========ERROR: Tham số <num_trainable> trong class PretrainedModel phải >= 0   ============="
            )

        # super(ConvNetBlock, self).__init__()
        super().__init__()
        self.name = name
        self.num_trainable = num_trainable
        self.model = None
        self.preprocess_input = None

    def build(self, input_shape):
        if self.name == "vgg16":
            self.model = keras.applications.vgg16.VGG16(
                weights="imagenet", include_top=False
            )
            self.preprocess_input = keras.applications.vgg16.preprocess_input
        elif self.name == "vgg19":
            self.model = keras.applications.vgg19.VGG19(
                weights="imagenet", include_top=False
            )
            self.preprocess_input = keras.applications.vgg19.preprocess_input
        else:
            raise ValueError(
                "=========ERROR: Pretrained model name is not valid============="
            )

        # Cập nhật trạng thái trainable cho các lớp đầu
        if self.num_trainable == 0:
            self.model.trainable = False
        else:
            self.model.trainable = True
            for layer in self.model.layers[: -self.num_trainable]:
                layer.trainable = False

    def call(self, x):

        x = self.preprocess_input(x)
        x = self.model(x)

        print("============= Đã qua được lớp PretrainedModel ===============")
        return x
