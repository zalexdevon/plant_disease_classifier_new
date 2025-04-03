from classifier.config.configuration import ConfigurationManager
from classifier.components.data_into_batches_splitter import (
    DataIntoBatchesSplitter,
)
from classifier import logger

STAGE_NAME = "DATA INTO BATCHES SPLITTING"


class DataIntoBatchesSplitterPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        config = config.get_data_into_batches_splitter_config()
        obj = DataIntoBatchesSplitter(config=config)

        try:
            obj.load_data_and_split()
            print(
                "============Splitting data into batches succesfully ====================="
            )
        except Exception as e:
            print(f"==========ERROR: =============\n{e}\n")


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIntoBatchesSplitterPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
