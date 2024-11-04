from BREASTCANCERCLASSIFICATION import logger
from BREASTCANCERCLASSIFICATION.pipeline.STAGE_01_DATA_INGESTION import DataIngestionTrainingPipeline
from BREASTCANCERCLASSIFICATION.pipeline.STAGE_02_PREPARE_BASE_MODEL import PrepareBaseModelTrainingPipeline
from BREASTCANCERCLASSIFICATION.pipeline.STAGE_03_TRAINING_PIPELINE import Model_Training_Pipeline

STAGE_NAME="DATA INGESTION STAGE"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started >>>>>>")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed >>>>>>")

except Exception as e:
    logger.info(e)
    raise e

STAGE_NAME="PREPARE BASE MODEL"

try:
    logger.info(f"**************")
    logger.info(f">>>>>>>{STAGE_NAME} started <<<<<<<<")
    prepare_base_model=PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>>>>>>>{STAGE_NAME} completed <<<<<<<<<<")

except Exception as e:
    raise e

STAGE_NAME="MODEL TRAINER"

try:
    logger.info(f"*************")
    logger.info(f">>>>>>>{STAGE_NAME} started <<<<<<<<")
    model_trainer=Model_Training_Pipeline()
    model_trainer.main()
    logger.info(f">>>>>>>{STAGE_NAME} Completed <<<<<<<<")
except Exception as e:
    raise e