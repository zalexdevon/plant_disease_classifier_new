from classifier.constants import *
from classifier.Mylib.myfuncs import read_yaml, create_directories
from classifier.entity.config_entity import (
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

    def get_model_trainer_config(
        self,
    ) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_trainer

        create_directories([config.root_dir, config.root_logs_dir])

        list_callbacks = [
            myfuncs.get_object_from_string_4(callback) for callback in params.callbacks
        ]

        list_layers = [
            myfuncs.get_object_from_string_4(layer) for layer in params.layers
        ]

        class_names = myfuncs.load_python_object(config.class_names_path)

        optimizer = myfuncs.get_object_from_string_4(params.optimizer)

        model_trainer_config = ModelTrainerConfig(
            # config
            train_ds_path=config.train_ds_path,
            val_ds_path=config.val_ds_path,
            class_names=class_names,
            best_model_path=config.best_model_path,
            results_path=config.results_path,
            structure_path=config.structure_path,
            detailed_structure_path=config.detailed_structure_path,
            list_monitor_components_path=config.list_monitor_components_path,
            # params
            is_first_time=params.is_first_time,
            model_name=params.model_name,
            epochs=params.epochs,
            callbacks=list_callbacks,
            layers_in_string=params.layers,
            layers=list_layers,
            optimizer=optimizer,
            loss=params.loss,
            metrics=params.metrics,
            # common params
            scoring=self.params.scoring,
            image_size=self.params.image_size,
            batch_size=self.params.batch_size,
        )

        return model_trainer_config

    # MODEL_EVALUATION
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        obj = ModelEvaluationConfig(
            test_data_path=config.test_data_path,
            preprocessor_path=config.preprocessor_path,
            model_path=config.model_path,
            result=config.result,
            target_col=config.target_col,
            metric=config.metric,
            evaluated_model_name=config.evaluated_model_name,
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
