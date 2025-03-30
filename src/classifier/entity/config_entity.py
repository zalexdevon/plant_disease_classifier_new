from dataclasses import dataclass
from pathlib import Path
from tensorflow.keras.optimizers import RMSprop


@dataclass(frozen=True)
class ModelTrainerConfig:
    # config
    train_ds_path: Path
    val_ds_path: Path
    class_names: list

    root_dir: Path
    root_logs_dir: Path
    best_model_path: Path
    results_path: Path
    structure_path: Path
    detailed_structure_path: Path

    # params
    is_first_time: str
    model_name: str
    epochs: int
    callbacks: list
    layers_in_string: str
    layers: list
    optimizer: RMSprop
    loss: str
    metrics: list

    # common params
    scoring: str
    image_size: int
    batch_size: int


# MODEL_EVALUATION
@dataclass(frozen=True)
class ModelEvaluationConfig:
    test_data_path: Path
    preprocessor_path: Path
    model_path: Path
    result: Path
    target_col: str
    metric: str
    evaluated_model_name: str


@dataclass(frozen=True)
class MonitorPlotterConfig:
    monitor_plot_html_path: Path
    monitor_plot_fig_path: Path
    target_val_value: float
    max_val_value: float
    dtick_y_value: float
