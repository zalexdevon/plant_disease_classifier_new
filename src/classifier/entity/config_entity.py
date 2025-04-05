from dataclasses import dataclass


# DATA INTO BATCHES SPLITTER
@dataclass(frozen=True)
class DataIntoBatchesSplitterConfig:
    # config input
    folder_path: str

    # config output
    root_dir: str
    class_names_path: str
    train_ds_path: str
    val_ds_path: str
    test_ds_path: str

    # params
    train_size: list
    val_size: int

    # common params
    image_size: int
    batch_size: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    # config input
    train_ds_path: str
    val_ds_path: str
    class_names_path: str

    # config output
    root_dir: str
    root_logs_dir: str
    best_models_in_training_dir: str
    best_model_path: str
    results_path: str
    model_structure_path: str
    list_monitor_components_path: str

    # params
    model_name: str
    epochs: int
    callbacks: list
    list_layers: list
    optimizer: str
    loss: str

    # common params
    scoring: str
    image_size: int
    batch_size: int
    metrics: list


# MODEL_EVALUATION
@dataclass(frozen=True)
class ModelEvaluationConfig:
    # config input
    test_ds_path: str
    class_names_path: str
    model_path: str

    # config output
    root_dir: str
    results_path: str

    # params
    metrics: list
    batch_size: int


@dataclass(frozen=True)
class MonitorPlotterConfig:
    monitor_plot_html_path: str
    monitor_plot_fig_path: str
    target_val_value: float
    max_val_value: float
    dtick_y_value: float
