import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')  #path where the preprocessor object will be saved   

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  #creating object of the class DataTransformationConfig    
    
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation based on different pipelines for numerical and categorical columns. It creates a preprocessor object that can be used to transform the data.   
        
        '''
        try:
            numerical_columns = ['writing_score', 'reading_score']  #numerical columns in the dataset
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']  #categorical columns in the dataset
            
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),  #imputing missing values with median
                ('scaler', StandardScaler())  #scaling the numerical columns
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  #imputing missing values with most frequent value
                ('one_hot_encoder', OneHotEncoder()),  #one hot encoding the categorical columns
                ('scaler', StandardScaler(with_mean=False))  #scaling the categorical columns
                ]
           )
            
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, numerical_columns),  #applying the numerical pipeline to the numerical columns
                ('cat_pipeline', cat_pipeline, categorical_columns)  #applying the categorical pipeline to the categorical columns
                ]
            )

            return preprocessor  #returning the preprocessor object

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)  #reading the train dataset
            test_df = pd.read_csv(test_path)  #reading the test dataset

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()  #getting the preprocessor object
            target_column_name = 'math_score'  #target column name
            numerical_columns = ['writing_score', 'reading_score']  #numerical columns in the dataset

            input_feature_train_df = train_df.drop(columns=[target_column_name])  #dropping the target column from the train dataset
            target_feature_train_df = train_df[target_column_name]  #getting the target column from the train dataset
            input_feature_test_df = test_df.drop(columns=[target_column_name])  #dropping the target column from the test dataset
            target_feature_test_df = test_df[target_column_name]  #getting the target column from the test dataset  

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
                )
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)  #fitting the preprocessor object on the train dataset  
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)  #transforming the test dataset using the preprocessor object 

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]  #combining the input features and target feature of the train dataset
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]  #combining the input features and target feature of the test dataset    

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )  #saving the preprocessor object

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,  #returning the path where the preprocessor object will be saved
            )

        except Exception as e:
            raise CustomException(e, sys)