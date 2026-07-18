#we will be having big data team collecting different sources storing different stores. aim is to read those
import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass    #you will be able to create a class with attributes without writing init method
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')  #artifacts folder will be created in the root directory
    test_data_path: str = os.path.join('artifacts', 'test.csv')  #artifacts folder will be created in the root directory
    raw_data_path: str = os.path.join('artifacts', 'data.csv')  #artifacts folder will be created in the root directory

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  #creating object of the class DataIngestionConfig

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')  #reading the data from the source
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)  #creating the artifacts folder if not exists
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)  #storing the raw data in the artifacts folder

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)  #splitting the data into train and test set

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)  #storing the train data in the artifacts folder
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)  #storing the test data in the artifacts folder

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()