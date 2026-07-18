#all the training code

import sys
from dataclasses import dataclass
import os

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging


from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")  #path where the trained model will be saved   

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()  #creating object of the class ModelTrainerConfig

    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):        
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  #input features of the train dataset
                train_array[:, -1],  #target feature of the train dataset
                test_array[:, :-1],  #input features of the test dataset
                test_array[:, -1],  #target feature of the test dataset
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)  #evaluating all the models and getting the r2 score of each model 


            for i in range(len(list(models))):  #iterating through all the models
                model = list(models.values())[i]  #getting the model object
                model.fit(X_train, y_train)  #fitting the model on the training data

                y_train_pred = model.predict(X_train)  #predicting on the training data
                y_test_pred = model.predict(X_test)  #predicting on the test data

                train_model_score = r2_score(y_train, y_train_pred)  #calculating r2 score on training data
                test_model_score = r2_score(y_test, y_test_pred)  #calculating r2 score on test data

                model_report[list(models.keys())[i]] = test_model_score  #storing the r2 score of each model in the dictionary

            best_model_score = max(sorted(model_report.values()))  #getting the best r2 score from all the models

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]  #getting the name of the best model

            best_model = models[best_model_name]
