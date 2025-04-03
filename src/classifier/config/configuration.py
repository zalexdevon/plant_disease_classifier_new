from classifier.constants import *
from classifier.Mylib.myfuncs import read_yaml, create_directories
from classifier.entity.config_entity import (
    DataIntoBatchesSplitterConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    MonitorPlotterConfig,
)
from pathlib import Path
from classifier.Mylib import myfuncs


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    # DATA INTO BATCHES SPLITTER
    def get_data_into_batches_splitter_config(self) -> DataIntoBatchesSplitterConfig:
        config = self.config.data_into_batches_splitter
        params = self.params.data_into_batches_splitter

        create_directories([config.root_dir])

        obj = DataIntoBatchesSplitterConfig(
            # config input
            folder_path=config.folder_path,
            # config output
            root_dir=config.root_dir,
            class_names_path=config.class_names_path,
            train_ds_path=config.train_ds_path,
            val_ds_path=config.val_ds_path,
            test_ds_path=config.test_ds_path,
            # params
            train_size=params.train_size,
            val_size=params.val_size,
            # common params
            image_size=self.params.image_size,
            batch_size=self.params.batch_size,
        )

        return obj

    def get_model_trainer_config(
        self,
    ) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_trainer

        create_directories(
            [config.root_dir, config.root_logs_dir, config.best_models_in_training_dir]
        )

        model_trainer_config = ModelTrainerConfig(
            # config input
            train_ds_path=config.train_ds_path,
            val_ds_path=config.val_ds_path,
            class_names_path=config.class_names_path,
            # config output
            root_dir=config.root_dir,
            root_logs_dir=config.root_logs_dir,
            best_models_in_training_dir=config.best_models_in_training_dir,
            best_model_path=config.best_model_path,
            results_path=config.results_path,
            model_structure_path=config.model_structure_path,
            list_monitor_components_path=config.list_monitor_components_path,
            # params
            is_first_time=params.is_first_time,
            model_name=params.model_name,
            epochs=params.epochs,
            callbacks=params.callbacks,
            model_training_type=params.model_training_type,
            list_layers=params.list_layers,
            optimizer=params.optimizer,
            loss=params.loss,
            # common params
            scoring=self.params.scoring,
            image_size=self.params.image_size,
            batch_size=self.params.batch_size,
            metrics=self.params.metrics,
        )

        return model_trainer_config

    # MODEL_EVALUATION
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        obj = ModelEvaluationConfig(
            # config input
            test_ds_path=config.test_ds_path,
            class_names_path=config.class_names_path,
            model_path=config.model_path,
            # config output
            root_dir=config.root_dir,
            results_path=config.results_path,
            # common params
            metrics=self.params.metrics,
            batch_size=self.params.batch_size,
        )

        return obj

    def get_monitor_plot_config(self) -> MonitorPlotterConfig:
        config = self.params.monitor_plotter

        obj = MonitorPlotterConfig(
            monitor_plot_html_path=config.monitor_plot_html_path,
            monitor_plot_fig_path=config.monitor_plot_fig_path,
            target_val_value=config.target_val_value,
            max_val_value=config.max_val_value,
            dtick_y_value=config.dtick_y_value,
        )

        return obj
