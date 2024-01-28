import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import *
from src.components.data_ingestion import DataIngestion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

@dataclass
class DataTranFormationConfig():
    preprocessor_obj_file_path = os.path.join("artifact", "preprocessor.pkl")
    

class DataTransformation():  
    def __init__(self) -> None:
        self.data_transformation_config = DataTranFormationConfig()      
    
    def get_data_transformer_obj(self):
        """
        This Function is resposible for data transformation based on various data types.
        """
        try:
            numrical_cols = ['reading_score', 'writing_score']
            categorical_cols = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            # Numerical Standard Scalar Pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Encoder Pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical Columns : {numrical_cols}")
            logging.info(f"Categorical Columns : {categorical_cols}")

            # Executing the Pipeline one by one
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numrical_cols),
                    ("categorical_pipeline", categorical_pipeline, categorical_cols)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_dataset = pd.read_csv(train_data_path)
            test_dataset = pd.read_csv(test_data_path)

            logging.info("Reading train and test data completed")

            logging.info("Obtaining the preprocessor Object")

            preprocessor_obj = self.get_data_transformer_obj()

            numrical_features = ['reading_score', 'writing_score']
            target_column = 'math_score'

            input_feature_train_dataset = train_dataset.drop(columns=[target_column], axis=1)
            target_feature_train_dataset = train_dataset[target_column]

            input_feature_test_dataset = test_dataset.drop(columns=[target_column], axis=1)
            target_feature_test_dataset = test_dataset[target_column]

            logging.info("Applying Preprocessor Object on training and testing dataset")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_dataset)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_dataset)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_dataset)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_dataset)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
            logging.info("Saved the preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)