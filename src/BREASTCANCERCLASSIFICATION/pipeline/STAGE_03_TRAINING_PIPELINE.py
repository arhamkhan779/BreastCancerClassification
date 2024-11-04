from BREASTCANCERCLASSIFICATION.config.configuration import ConfigurationManager
from BREASTCANCERCLASSIFICATION.components.MODEL_TRAINING import Training
from BREASTCANCERCLASSIFICATION import logger

STAGE_NAME="MODEL TRAINER"

class Model_Training_Pipeline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()
        training_config=config.get_training_config()
        obj=Training(training_config)
        model=obj.get_base_model()
        X,Y=obj.create_data_set()
        obj.Training(X,Y,model)