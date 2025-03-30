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
from sklearn.linear_model import LogisticRegression
import ast
from io import StringIO
import sys
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Rescaling,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
)

from keras_cv.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomBrightness,
    RandomGaussianBlur,
    RandomContrast,
    RandomHue,
    RandomSaturation,
)


from tensorflow.keras.optimizers import RMSprop


def get_sum(a, b):
    """Demo function for the library"""
    return a + b


def get_outliers(data):
    """Lấy các giá trị outlier nằm ngoài khoảng Q1 - 1.5*IQR và Q3 + 1.5*IQR
    Args:
        data (_type_): một mảng các số

    Returns:
        _type_: các số outliers
    """

    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return outliers


@ensure_annotations
def get_exact_day(seconds_since_epoch: int):
    """Get the exact day from 1/1/1970

    Args:
        seconds_since_epoch (int): seconds

    Returns:
        datetime: the exact day
    """

    epoch = datetime(1970, 1, 1)
    return epoch + timedelta(seconds=seconds_since_epoch)


@ensure_annotations
def is_number(string_to_check: str):
    """Check if string_to_check is a number"""

    try:
        float(string_to_check)
        return True
    except ValueError:
        return False


@ensure_annotations
def is_integer_str(s: str):
    """Check if str is an integer

    Args:
        s (str): _description_
    """

    regex = "^[+-]?\d+$"
    return re.match(regex, s) is not None


@ensure_annotations
def is_natural_number_str(s: str):
    """Check if str is a natural_number

    Args:
        s (str): _description_
    """

    regex = "^\+?\d+$"
    return re.match(regex, s) is not None


@ensure_annotations
def is_natural_number(num: numbers.Real):
    """Check if num is a natural number

    Args:
        num (numbers.Real): _description_

    """

    return is_integer(num) and num >= 0


@ensure_annotations
def is_integer(number: numbers.Real):
    """Check if number is a integer

    Args:
        number (numbers.Real): _description_
    """

    return pd.isnull(number) == False and number == int(number)


def get_combinations_k_of_n(arr, k):
    """Get the combinations k of arr having n elements"""

    return list(itertools.combinations(arr, k))


def show_frequency_table(arr):
    """Show the frequency table of arr"""
    counts, bin_edges = np.histogram(arr, bins="auto")

    frequency_table = pd.DataFrame(
        {"Bin Start": bin_edges[:-1], "Bin End": bin_edges[1:], "Count": counts}
    )

    frequency_table["Percent"] = frequency_table["Count"] / len(arr) * 100
    frequency_table["Percent"] = frequency_table["Percent"].round(2)

    return frequency_table


def extract_zip_file(zip_file_path, unzip_path):
    """Extract zip file

    Args:
        zip_file_path (str): file path of zip file
        unzip_path (str): folder path of unzip components
    """

    with ZipFile(zip_file_path, "r") as zip:
        zip.extractall(path=unzip_path)


def create_sub_folder_from_dataset(path, data_proportion, root_dir):
    """
    Args:
        path: the path to the folder to create
        root_dir : the path to the folder Dataset
        data_proportion: the proportion of taken data
    Returns:
        _str_: result
    Examples:
        vd tạo thư mục train là 70% dữ liệu, thư mục val là 15% dữ liệu, thư mục test là 15% dữ liệu

        lưu ý là di chuyển các ảnh chứ không phải copy

        nên tập val lấy 0.5 = 0.5 đối với dữ liệu còn lại

        Code:
        ```python
        dataset_path = './Dataset'

        path = './train'
        create_sub_folder_from_dataset(path, 0.7, dataset_path)

        path = './val'
        create_sub_folder_from_dataset(path, 0.5, dataset_path)

        path = './test'
        create_sub_folder_from_dataset(path, 1, dataset_path)
        ```
    """

    if not os.path.exists(path):
        os.mkdir(path)

        for dir in os.listdir(root_dir):
            os.makedirs(os.path.join(path, dir))

            img_names = np.random.permutation(os.listdir(os.path.join(root_dir, dir)))
            count_selected_img_names = int(data_proportion * len(img_names))
            selected_img_names = img_names[:count_selected_img_names]

            for img in selected_img_names:
                src = os.path.join(root_dir, dir, img)
                dest = os.path.join(path, dir)
                shutil.move(src, dest)

        return "Create the sub-folder successfully"
    else:
        return "The sub-folder existed"


def fetch_source_url_to_zip_file(source_url, local_zip_path):
    """Download file from url to local

    Returns:
        filename: the name of file
        headers: info about the file
    """

    os.makedirs(local_zip_path, exist_ok=True)
    filename, headers = urllib.request.urlretrieve(source_url, local_zip_path)

    return filename, headers


@ensure_annotations
def split_numpy_array(
    data: np.ndarray, ratios: list, dimension=1, shuffle: bool = True
):
    """

    Args:
        data (np.ndarray): _description_
        ratios (list): Tỉ lệ các phần. Tổng phải bằng 1
        dimension (int, optional): Chiều của dữ liệu. nếu dữ liệu 2 chiều thì gán = 2. Defaults to 1.
        shuffle(bool, optional): có xáo trộn dữ liệu trước khi chia không

    Returns:
        list: list các mảng numpy

    vd:
    với dữ liệu 2 chiều:
    ```python
    split_ratios = [0.5, 0.2, 0.2, 0.1]  # Tỷ lệ mong muốn
    subsets = split_data(data, split_ratios, 2)
    ```
    """
    if sum(ratios) != 1:
        raise ValueError("Tổng của ratios phải bằng 1")

    if shuffle:
        data = np.random.permutation(data)

    len_data = len(data) if dimension == 1 else data.shape[0]
    split_indices = np.cumsum(ratios)[:-1] * len_data
    split_indices = split_indices.astype(int)
    return (
        np.split(data, split_indices)
        if dimension == 1
        else np.split(data, split_indices, axis=0)
    )


@ensure_annotations
def split_dataframe_data(data: pd.DataFrame, ratios: list, shuffle: bool = True):
    """

    Args:
        data (pd.DataFrame): _description_
        ratios (list): Tỉ lệ các phần. Tổng phải bằng 1
        shuffle(bool, optional): có xáo trộn dữ liệu trước khi chia không. Defaults to True

    Returns:
        list: list các dataframe

    VD:
        ```python
    split_ratios = [0.5, 0.2, 0.2, 0.1]  # Tỷ lệ mong muốn
    subsets = split_data(data, split_ratios)
    """
    if sum(ratios) != 1:
        raise ValueError("Tổng của ratios phải bằng 1")

    if shuffle:
        data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

    split_indices = np.cumsum(ratios)[:-1] * len(data)
    split_indices = split_indices.astype(int)

    subsets = np.split(data, split_indices, axis=0)

    return [pd.DataFrame(item, columns=data.columns) for item in subsets]


def load_python_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise e


def save_python_object(file_path, obj):
    """Save python object in a file

    Args:
        file_path (_type_): ends with .pkl
    """

    try:

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise e


def np_arange_int(start, end, step):
    """Tạo ra dãy số nguyên cách nhau step"""

    return np.arange(start, end + step, step)


def np_arange_float(start, end, step):
    """Tạo ra dãy số thực cách nhau step"""

    return np.arange(start, end, step)


def np_arange(start, end, step):
    """Create numbers from start to end with step, used for both **float** and **int** number <br>
    used for int: start, end, step must be int <br>
    used for float: start must be float <br>
    """

    if is_integer(start) and is_integer(end) and is_integer(step):
        return np_arange_int(int(start), int(end), int(step))

    return np_arange_float(start, end, step)


def get_range_for_param(param_str):
    """Create values range from param_str

    VD:
        param_str = format=start-end-step 12-15-1
        param_str = format=num 12
        param_str = format=start-end start, mean, end vd: 12-15 -> 12 13 15



    """
    if "-" not in param_str:
        if is_integer_str(param_str):
            return [int(param_str)]

        return [float(param_str)]

    if param_str.count("-") == 2:
        nums = param_str.split("-")
        num_min = float(nums[0])
        num_max = float(nums[1])
        num_step = float(nums[2])

        return np_arange(num_min, num_max, num_step)

    nums = param_str.split("-")
    num_min = float(nums[0])
    num_max = float(nums[1])

    num_mean = None
    if is_integer(num_min) and is_integer(num_max):
        num_min = int(num_min)
        num_max = int(num_max)

        num_mean = int((num_min + num_max) / 2)
    else:
        num_mean = (num_min + num_max) / 2

    return [num_min, num_mean, num_max]


@ensure_annotations
def generate_grid_search_params(param_grid: dict):
    """Generate all combinations of params like grid search

    Returns:
        list:
    """

    keys = param_grid.keys()
    values = (param_grid[key] for key in keys)
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def get_num_de_cac_combinations(list_of_list):
    """Count the number of De cac combinations of list_of_list, which is the list of list"""

    return math.prod(map(len, list_of_list))


@ensure_annotations
def read_yaml(path_to_yaml: str) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (Path): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: _description_
    """
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            print(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        verbose (bool, optional): ignore if multiple dirs is to be created. Defaults to True.

    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"json file saved at {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    print(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    print(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    print(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size_inKB(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())


def unindent_all_lines(content):
    content = content.strip("\n")
    lines = content.split("\n")
    lines = [item.strip() for item in lines]
    content_processed = "\n".join(lines)

    return content_processed


def insert_br_html_at_the_end_of_line(lines):
    return [f"{item} <br>" for item in lines]


def get_monitor_desc(param_grid_model_desc: dict):
    result = ""

    for key, value in param_grid_model_desc.items():
        key_processed = process_param_name(key)
        line = f"{key_processed}: {value}<br>"
        result += line

    return result


def process_param_name(name):
    """if param name is max_depth -> max

    if param name is C -> C\_\_

    """

    if len(name) == 3:
        return name

    if len(name) > 3:
        return name[:3]

    return name + "_" * (3 - len(name))


def get_param_grid_model(param_grid_model: dict):
    """Get param grid from file params.yaml

    VD:
    ```python
    In params.yaml
    param_grid_model_desc:
      alpha: 0.2-0.7-0.1
      l1_ratio: 0.2-0.4

    convert to
    param_grid = {
        "alpha": np.arange(0.2, 0.7, 0.1)
        "l1_ratio": [0.2, 0.3, 0.4]
    }
    ``
    """

    values = param_grid_model.values()

    values = [get_range_for_param(str(item)) for item in values]

    return dict(zip(list(param_grid_model.keys()), values))


def sub_param_for_yaml_file(src_path: str, des_path: str, replace_dict: dict):
    """Substitue params in src_path and save in des_path

    Args:
        replace_dict (dict): key: item needed to replace, value: item to replace
        VD:
        ```python
        replace_dict = {
            "${P}": data_transformation,
            "${T}": model_name,
            "${E}": evaluation,
        }

        ```
    """

    with open(src_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    config_str = yaml.dump(config_data, default_flow_style=False)

    for key, value in replace_dict.items():
        config_str = config_str.replace(key, value)

    with open(des_path, "w", encoding="utf-8") as file:
        file.write(config_str)

    print(f"Đã thay thế các tham số trong {src_path} lưu vào {des_path}")


def get_real_column_name(column):
    """After using ColumnTransformer, the column name has format = bla__Age, so only take Age"""

    start_index = column.find("__") + 2
    column = column[start_index:]
    return column


def get_real_column_name_from_get_feature_names_out(columns):
    """Take the exact name from the list retrieved by method get_feature_names_out() of ColumnTransformer"""

    return [get_real_column_name(item) for item in columns]


def fix_name_by_LGBM_standard(cols):
    """LGBM standard state that columns name can only contain characters among letters, digit and '_'

    Returns:
        list: _description_
    """

    cols = pd.Series(cols)
    cols = cols.str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    return list(cols)


def find_feature_importances(train_data, model):
    """Find the feature importances of some models like: RF, GD, SGD, LGBM

    Returns:
        pd.DataFrame:
    """

    score = pd.DataFrame(
        data={
            "feature": train_data.columns.tolist(),
            "score": model.feature_importances_,
        }
    )
    score = score.sort_values(by="score", ascending=False)
    return score


def find_coef_with_classifier(train_data, model):
    """Find the feature importances of some models like: LR1, LRe

    Returns:
        pd.DataFrame:
    """

    score = pd.DataFrame(
        data={
            "feature": train_data.columns.tolist(),
            "score": np.abs(model.coef_[0]),
        }
    )
    score = score.sort_values(by="score", ascending=False)
    return score


def get_params_transform_list_to_1_value(param_grid):
    """Create params with key and one value not a list with one value

    Args:
        param_grid (_type_): value is a list with only one value

    Returns:
        dict: _description_

    VD:
    ```python
    param_grid = {
    "C": [1],
    "A": [2],
    }

    Convert to

    param_grid = {
    "C": 1,
    "A": 2,
    }
    ```
    """

    values = [item[0] for item in param_grid.values()]
    return dict(zip(param_grid.keys(), values))


def get_describe_stats_for_numeric_cat_cols(data):
    """Get descriptive statistics of numeric cat cols, including min, max, median

    Args:
        data (_type_): numeric cat cols
    Returns:
        Dataframe: min, max, median
    """

    min_of_cols = data.min().to_frame(name="min")
    max_of_cols = data.max().to_frame(name="max")
    median_of_cols = data.quantile([0.5]).T.rename(columns={0.5: "median"})

    result = pd.concat([min_of_cols, max_of_cols, median_of_cols], axis=1).T

    return result


def split_tfdataset_into_tranvaltest_1(
    ds: tf.data.Dataset,
    train_split=0.8,
    val_split=0.1,
    shuffle=True,
    shuffle_size=10000,
):
    """Chia dataset thành tập train, val, test theo tỉ lệ nhất định

    Args:
        ds (tf.data.Dataset): _description_
        train_split (float, optional): _description_. Defaults to 0.8.
        val_split (float, optional): _description_. Defaults to 0.1.
        shuffle (bool, optional): _description_. Defaults to True.
        shuffle_size (int, optional): _description_. Defaults to 10000.

    Returns:
        train, val, test
    """
    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=42)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def cache_prefetch_tfdataset_2(ds: tf.data.Dataset, shuffle_size=1000):
    return ds.cache().shuffle(shuffle_size).prefetch(buffer_size=tf.data.AUTOTUNE)


def train_test_split_tfdataset_3(
    ds: tf.data.Dataset, test_size=0.2, shuffle=True, shuffle_size=10000
):
    """Chia dataset thành tập train, test theo tỉ lệ của tập test

    Returns:
        _type_: train_ds, test_ds
    """
    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=42)

    test_size = int(test_size * ds_size)

    test_ds = ds.take(test_size)
    train_ds = ds.skip(test_size)

    return train_ds, test_ds


def get_object_from_string_4(text: str):
    """Get đối tượng từ 1 chuối

    Example:
        text = "LogisticRegression(C=144, penalty=l1, solver=saga,max_iter=10000,dual=True)"

        -> đối tượng LogisticRegression(C=144, dual=True, max_iter=10000, penalty='l1',solver='saga')

    Args:
        text (str): _description_


    """
    # Tách tên lớp và tham số
    class_name, params = text.split("(", 1)
    params = params[:-1]

    # TODO: d
    print("Class = " + class_name)
    print("params = " + params)
    # d

    object_class = globals()[class_name]

    if params == "":
        return object_class()

    # Lấy tham số của đối tượng
    param_parts = params.split(",")
    param_parts = [item.strip() for item in param_parts]
    keys = [item.split("=")[0].strip() for item in param_parts]

    values = [
        do_ast_literal_eval_advanced_7(item.strip().split("=")[1].strip())
        for item in param_parts
    ]

    params = dict(zip(keys, values))

    return object_class(**params)


def get_object_from_string_using_eval_6(text: str, module):
    """Get đối tượng từ 1 chuối

    Example:
        text = 'LogisticRegression(C=144,penalty="l1",solver="saga",max_iter=10000,dual=True)' -> các chuỗi phải được bọc trong cặp ""

        module = sklearn.linear_model

        -> đối tượng LogisticRegression(C=144, dual=True, max_iter=10000, penalty='l1',solver='saga')

    Args:
        text (str): _description_


    """

    # Tách tên lớp và tham số
    class_name, params = text.split("(", 1)
    params = params.rstrip(")")

    class_name = getattr(module, class_name)

    # Tạo đối tượng bằng eval (không khuyến khích nếu có dữ liệu không đáng tin cậy)
    return eval(f"class_name({params})")


def do_ast_literal_eval_advanced_7(text: str):
    """Kế thừa hàm ast.literal_eval() nhưng xử lí thêm trường hợp sau

    Tuple dạng (1.0 ; 2.0), các phần tử cách nhau bởi dấu ; thay vì dấu ,
    """
    if ";" not in text:
        return ast.literal_eval(text)

    items = text.strip().split(";")
    return (ast.literal_eval(item) for item in items)
