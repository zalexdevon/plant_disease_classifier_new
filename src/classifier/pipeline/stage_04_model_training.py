from classifier.config.configuration import ConfigurationManager
from classifier.components.model_trainer import (
    ModelTrainer,
)
from classifier import logger
from classifier.components.monitor_plotter import (
    MonitorPlotter,
)

STAGE_NAME = "Model Training stage"


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)

        try:
            model_trainer.load_data_to_train()
            print("\n===== Load data thành công ====== \n")
            model_trainer.load_callbacks()
            print("\n===== Load callbacks thành công ====== \n")
            model_trainer.load_model()
            print("\n===== Load model thành công ====== \n")
            model_trainer.train()
            print("\n===== Train thành công ====== \n")
            model_trainer.save_model()
            print("\n===== Save model thành công ====== \n")
            model_trainer.save_list_monitor_components()
            print("\n===== Save kết quả các lần chạy model thành công ====== \n")

            monitor_plotter_config = config.get_monitor_plot_config()
            monitor_plotter = MonitorPlotter(monitor_plotter_config)
            monitor_plotter.plot(model_trainer.list_monitor_components)

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
