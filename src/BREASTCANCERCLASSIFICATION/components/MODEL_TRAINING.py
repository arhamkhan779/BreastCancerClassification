from BREASTCANCERCLASSIFICATION.entity.config_entity import TrainingConfig
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from BREASTCANCERCLASSIFICATION.config.configuration import ConfigurationManager
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from BREASTCANCERCLASSIFICATION import logger
import pandas as pd

class Training:
    def __init__(self,config:TrainingConfig):
        self.config=config

    def get_base_model(self):
        self.model=tf.keras.models.load_model(
            self.config.update_base_model_path
        )
        self.optimizer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer=self.optimizer, loss='SparseCategoricalCrossentropy', metrics=['accuracy'])
        return self.model

    def create_data_set(self):

        logger.info("Daataset Loading --------- Start")
        self.Y=[]
        self.X=[]
        self.labels_dict={}
        self.images_dict={}
        count=0
        self.Dataset_path=self.config.training_data
        self.windows_path=Path(self.Dataset_path)

        for classes in os.listdir(self.Dataset_path):
            self.labels_dict[classes]=count
            count+=1
        
        self.benign=list(self.windows_path.glob('benign/*'))
        self.melignant=list(self.windows_path.glob('malignant/*'))
        self.normal=list(self.windows_path.glob('normal/*'))

        self.images_dict['benign']=self.benign
        self.images_dict['malignant']=self.melignant
        self.images_dict['normal']=self.normal

        try:
            for flower_name ,flower_images in self.images_dict.items():
                for images in flower_images:
                    img=cv2.imread(str(images))
                    img=cv2.resize(img,(224,224))
                    img=img/255
                    self.X.append(img)
                    self.Y.append(self.labels_dict[flower_name])
            self.X=np.array(self.X)
            self.Y=np.array(self.Y)
            logger.info("Dataset Loading ------- Completed")
            return self.X,self.Y  
        except Exception as e:
            logger.info(e)
            raise e      
    
    def Training(self,X,Y,model):
        logger.info("splitting Data into Training and Validation")
        try:
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=45)
            logger.info(f"Start Training ON Epochs :{self.config.params_epoch} Image Size :{self.config.params_img_size} Batch :{self.config.params_batch_size}")
            self.training_history=model.fit(X_train,Y_train,epochs=self.config.params_epoch
                                        ,batch_size=self.config.params_batch_size,validation_data=(X_test,Y_test))
            logger.info(f"Saving Training and Validation results at {self.config.results}")
            self.history_frame=pd.DataFrame(self.training_history.history)
            self.history_frame.to_csv(self.config.results,index=False)
            logger.info(f"Saving Model at {self.config.trained_model_path}")
            model.save(self.config.trained_model_path)
        except Exception as e:
            logger.info(e)
            raise e

    
    
    