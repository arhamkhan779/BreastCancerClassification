from BREASTCANCERCLASSIFICATION.config.configuration import ConfigurationManager
from BREASTCANCERCLASSIFICATION.components.PREPARE_BASE_MODEL import PrepareBaseModel
from BREASTCANCERCLASSIFICATION import logger

STAGE_NAME="PREPARE BASE MODEL"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()
        prepare_base_model_config=config.get_prepare_base_model_config()
        prepare_base_model=PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.save_updated_model()

    