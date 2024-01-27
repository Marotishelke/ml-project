# Import the libararies
import os
import sys 
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig():
    train_data_path : str = os.path.join("artifact", "train_data.csv")
    test_data_path : str = os.path.join("artifact", "test_data.csv")
    raw_data_path : str = os.path.join("artifact", "raw_data.csv")

class DataIngestion():
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component!")

        try:
            dataset = pd.read_csv("./notebook/data/stud.csv")
            logging.info("Succefully read the dataset and store as data frame")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)

            dataset.to_csv(self.data_ingestion_config.raw_data_path, index=None, header=True)

            logging.info("Train Test Split Intiated...")
            train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=143)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index=None, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=None, header=True)

            logging.info("Ingestion of data is completed.")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
    
if __name__ == '__main__':
    data_ingestion_obj = DataIngestion()
    data_ingestion_obj.initiate_data_ingestion()