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
import matplotlib.cm as cm


class ConvNetBlock_XceptionVersion(layers.Layer):
    """Gồm các layers sau:
    - SeparableConv2D
    - SeparableConv2D
    - MaxPooling2D

    Đi kèm là sự kết hợp giữa residual connections, batch normalization và  depthwise separable convolutions (lớp SeparableConv2D)

    Attributes:
        filters (_type_): số lượng filters trong lớp SeparableConv2D
    """

    def __init__(self, filters, name=None, **kwargs):
        """_summary_"""
        # super(ConvNetBlock_XceptionVersion, self).__init__()
        super().__init__(name=name, **kwargs)
        self.filters = filters

    def build(self, input_shape):

        self.BatchNormalization = layers.BatchNormalization()
        self.Activation = layers.Activation("relu")
        self.SeparableConv2D = layers.SeparableConv2D(
            self.filters, 3, padding="same", use_bias=False
        )

        self.BatchNormalization_1 = layers.BatchNormalization()
        self.Activation_1 = layers.Activation("relu")
        self.SeparableConv2D_1 = layers.SeparableConv2D(
            self.filters, 3, padding="same", use_bias=False
        )
        self.MaxPooling2D = layers.MaxPooling2D(3, strides=2, padding="same")

        self.Conv2D = layers.Conv2D(
            self.filters, 1, strides=2, padding="same", use_bias=False
        )

        super().build(input_shape)

    def call(self, x):
        residual = x

        # First part of the block
        x = self.BatchNormalization(x)
        x = self.Activation(x)
        x = self.SeparableConv2D(x)

        # Second part of the block
        x = self.BatchNormalization_1(x)
        x = self.Activation_1(x)
        x = self.SeparableConv2D_1(x)
        x = self.MaxPooling2D(x)

        # Apply residual connection
        residual = self.Conv2D(residual)
        x = layers.add([x, residual])

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        name = config.pop("name", None)
        return cls(**config)


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

    def __init__(self, filters, pooling=True, **kwargs):
        # super(ConvNetBlock_Advanced, self).__init__()
        super().__init__(**kwargs)
        self.filters = filters
        self.pooling = pooling

    def build(self, input_shape):

        self.BatchNormalization = layers.BatchNormalization()
        self.Activation = layers.Activation("relu")
        self.Conv2D = layers.Conv2D(self.filters, 3, padding="same")

        self.BatchNormalization_1 = layers.BatchNormalization()
        self.Activation_1 = layers.Activation("relu")
        self.Conv2D_1 = layers.Conv2D(self.filters, 3, padding="same")

        self.MaxPooling2D = layers.MaxPooling2D(2, padding="same")

        self.Conv2D_2 = layers.Conv2D(self.filters, 1, strides=2)

        self.Conv2D_3 = layers.Conv2D(self.filters, 1)

        super().build(input_shape)

    def call(self, x):
        residual = x

        # First part of the block
        x = self.BatchNormalization(x)
        x = self.Activation(x)
        x = self.Conv2D(x)

        # Second part of the block
        x = self.BatchNormalization_1(x)
        x = self.Activation_1(x)
        x = self.Conv2D_1(x)

        # Có dùng MaxPooling2D không
        if self.pooling:
            x = self.MaxPooling2D(x)
            residual = self.Conv2D_2(
                residual
            )  # Có dùng max pooling thì kèm cái này thôi
        elif self.filters != residual.shape[-1]:
            residual = self.Conv2D_3(
                residual
            )  # Nếu không dùng max pooling thì kèm cái này

        # Apply residual connection
        x = layers.add([x, residual])

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "pooling": self.pooling,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        name = config.pop("name", None)
        return cls(**config)


class ConvNetBlock(layers.Layer):
    """Kết hợp các layers sau:
    - Conv2D * num_Conv2D_layer
    - MaxPooling

    Attributes:
        filters (_type_): số lượng filters trong lớp Conv2D
        num_Conv2D (int, optional): số lượng lớp num_Conv2D. Defaults to 1.
    """

    def __init__(self, filters, num_Conv2D=1, name=None, **kwargs):
        """ """
        # super(ConvNetBlock, self).__init__()
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.num_Conv2D = num_Conv2D

    def build(self, input_shape):
        self.list_Conv2D = [
            layers.Conv2D(self.filters, 3, activation="relu")
            for _ in range(self.num_Conv2D)
        ]

        self.MaxPooling2D = layers.MaxPooling2D(pool_size=2)

        super().build(input_shape)

    def call(self, x):
        for conv2D in self.list_Conv2D:
            x = conv2D(x)

        x = self.MaxPooling2D(x)

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "num_Conv2D": self.num_Conv2D,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # TODO: d
    def print(self):
        print("Ok roi !!!!!!")

    # d


class ImageDataPositionAugmentation(layers.Layer):
    """Tăng cường dữ liệu hình ảnh ở khía cạnh vị trí, bao gồm các lớp sau (**trong tf.keras.layers**)
    - RandomFlip
    - RandomRotation
    - RandomZoom

    Attributes:
        rotation_factor (float): Tham số cho lớp RandomRotation. Default to 0.2
        zoom_factor (float): Tham số cho lớp RandomZoom. Default to 0.2
    """

    def __init__(self, rotation_factor=0.2, zoom_factor=0.2, **kwargs):
        # super(ImageDataPositionAugmentation, self).__init__()
        super().__init__(**kwargs)
        self.rotation_factor = rotation_factor
        self.zoom_factor = zoom_factor

    def build(self, input_shape):
        self.RandomFlip = layers.RandomFlip(mode="horizontal_and_vertical")
        self.RandomRotation = layers.RandomRotation(factor=self.rotation_factor)
        self.RandomZoom = layers.RandomZoom(height_factor=self.zoom_factor)

        super().build(input_shape)

    def call(self, x):
        x = self.RandomFlip(x)
        x = self.RandomRotation(x)
        x = self.RandomZoom(x)

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "rotation_factor": self.rotation_factor,
                "zoom_factor": self.zoom_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        name = config.pop("name", None)
        return cls(**config)


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
        **kwargs
    ):
        # super(ImageDataColorAugmentation, self).__init__()
        super().__init__(**kwargs)
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.hue_factor = hue_factor
        self.saturation_factor = saturation_factor

    def build(self, input_shape):
        self.RandomBrightness = keras_cv.layers.RandomBrightness(
            factor=self.brightness_factor
        )
        self.RandomGaussianBlur = keras_cv.layers.RandomGaussianBlur(
            kernel_size=3, factor=(0.0, 1.0)
        )
        self.RandomContrast = keras_cv.layers.RandomContrast(
            factor=self.contrast_factor,
            value_range=(1 - self.contrast_factor, 1 + self.contrast_factor),
        )
        self.RandomHue = keras_cv.layers.RandomHue(
            factor=self.hue_factor,
            value_range=(1 - self.hue_factor, 1 + self.hue_factor),
        )
        self.RandomSaturation = keras_cv.layers.RandomSaturation(
            factor=self.saturation_factor
        )

        super().build(input_shape)

    def call(self, x):
        x = self.RandomBrightness(x)
        x = self.RandomGaussianBlur(x)  # Lớp này để mặc định
        x = self.RandomContrast(x)
        x = self.RandomHue(x)
        x = self.RandomSaturation(x)

        return x

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh, bao gồm cả tham số trainable và dtype
        config = super().get_config()
        config.update(
            {
                "brightness_factor": self.brightness_factor,
                "contrast_factor": self.contrast_factor,
                "hue_factor": self.hue_factor,
                "saturation_factor": self.saturation_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Loại bỏ tham số 'name' từ config (vì Keras đã xử lý nó)
        name = config.pop("name", None)
        return cls(**config)


class PretrainedModel(layers.Layer):
    """Sử dụng các pretrained models ở trong **keras.applications**
    Attributes:
        model_name (str): Tên pretrained model, vd: vgg16, vgg19, ....
        num_trainable (int, optional): Số lượng các lớp đầu tiên cho trainable = True. Defaults to 0.
    """

    def __init__(self, model_name, num_trainable=0, **kwargs):
        if num_trainable < 0:
            raise ValueError(
                "=========ERROR: Tham số <num_trainable> trong class PretrainedModel phải >= 0   ============="
            )

        # super(ConvNetBlock, self).__init__()
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_trainable = num_trainable

    def build(self, input_shape):
        if self.model_name == "vgg16":
            self.model = keras.applications.vgg16.VGG16(
                weights="imagenet", include_top=False
            )
            self.preprocess_input = keras.applications.vgg16.preprocess_input
        elif self.model_name == "vgg19":
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

        super().build(input_shape)

    def call(self, x):
        x = self.preprocess_input(x)
        x = self.model(x)

        return x


class GradCAMForImages:
    """Thực hiện quá trình GradCAM để xác định những phần nảo của ảnh hỗ trợ model phân loại nhiều nhất
    Attributes:
        images (_type_): Tập ảnh đã được chuyển thành **array**
        model (_type_): model
        last_convnet_layer_name (_type_): **Tên** hoặc  **index** của layer convent cuối cùng trong model
    """

    def __init__(self, images, model, last_convnet_layer_name):
        self.images = images
        self.model = model
        self.last_convnet_layer_name = last_convnet_layer_name

    def create_models(self):
        """Tạo ra 2 model sau:

        **last_conv_layer_model**: model map input image -> convnet block cuối cùng

        **classifier_model**: model map convnet block cuối cùng -> final class predictions.

        Returns:
            tuple: last_conv_layer_model, classifier_model
        """
        last_conv_layer = None
        classifier_layers = None
        if isinstance(self.last_convnet_layer_name, str):
            layer_names = [layer.name for layer in self.model.layers]
            last_conv_layer = self.model.get_layer(self.last_convnet_layer_name)
            classifier_layers = self.model.layers[
                layer_names.index(self.last_convnet_layer_name) + 1 :
            ]
        else:
            last_conv_layer = self.model.layers[self.last_convnet_layer_name]
            classifier_layers = self.model.layers[self.last_convnet_layer_name + 1 :]

        # Model đầu tiên
        last_conv_layer_model = keras.Model(
            inputs=self.model.inputs, outputs=last_conv_layer.output
        )

        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input

        for layer in classifier_layers:
            x = layer(x)

        # Model thứ hai
        classifier_model = keras.Model(inputs=classifier_input, outputs=x)

        return last_conv_layer_model, classifier_model

    def do_gradient(self, last_conv_layer_model, classifier_model):
        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(self.images)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        return grads, last_conv_layer_output

    def get_heatmap(self, grads, last_conv_layer_output):
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        last_conv_layer_output = last_conv_layer_output.numpy()[0]

        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(last_conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

    def convert_1image(self, img, heatmap):
        heatmap = np.uint8(255 * heatmap)

        jet = cm.get_cmap("jet")  # Dùng "jet" để tô màu lại heatmap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * 0.4 + img
        superimposed_img = keras.utils.array_to_img(superimposed_img)

        return superimposed_img

    def convert(self):
        last_conv_layer_model, classifier_model = self.create_models()
        grads, last_conv_layer_output = self.do_gradient(
            last_conv_layer_model, classifier_model
        )
        heatmap = self.get_heatmap(grads, last_conv_layer_output)

        list_superimposed_img = [
            self.convert_1image(img, heatmap) for img in self.images
        ]

        return list_superimposed_img


class ImagesToArrayConverter:
    """Chuyển 1 tập ảnh thành 1 mảng numpy

    Attributes:
        image_paths (list): Tập các đường dẫn đến các file ảnh
        target_size (int): Kích thước sau khi resize
    """

    def __init__(self, image_paths, target_size):

        self.image_paths = image_paths
        self.target_size = (target_size, target_size)

    def convert_1image(self, img_path):
        img = keras.utils.load_img(
            img_path, target_size=self.target_size
        )  # load ảnh và resize luôn
        array = keras.utils.img_to_array(img)  # Chuyển img sang array
        array = np.expand_dims(
            array, axis=0
        )  # Thêm chiều để tạo thành mảng có 1 phần tử
        return array

    def convert(self):
        return np.vstack(
            [self.convert_1image(img_path) for img_path in self.image_paths]
        )
