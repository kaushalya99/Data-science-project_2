import os
import pickle
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)  #getting the directory path of the file
        os.makedirs(dir_path, exist_ok=True)  #creating the directory if it doesn't exist

        with open(file_path, "wb") as file_obj:  #opening the file in write binary mode
            dill.dump(obj, file_obj)  #saving the object to the file
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):

    try:
        report = {}  #to store the r2 score of each model

        for model_name, model in models.items():  #iterating through all the models
            model_params = param.get(model_name, {})  #getting the hyperparameters of the model

            if model_params:
                gs = GridSearchCV(model, model_params, cv=3)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)  #fitting the model on the training data

            y_train_pred = model.predict(X_train)  #predicting the target feature of the train dataset
            y_test_pred = model.predict(X_test)  #predicting the target feature of the test dataset

            train_model_score = r2_score(y_train, y_train_pred)  #calculating the r2 score of the model on the train dataset
            test_model_score = r2_score(y_test, y_test_pred)  #calculating the r2 score of the model on the test dataset

            report[model_name] = test_model_score  #storing the r2 score of the model in the report dictionary

        return report  #returning the report dictionary

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)