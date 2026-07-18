import os
import pickle
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)  #getting the directory path of the file
        os.makedirs(dir_path, exist_ok=True)  #creating the directory if it doesn't exist

        with open(file_path, "wb") as file_obj:  #opening the file in write binary mode
            dill.dump(obj, file_obj)  #saving the object to the file
    except Exception as e:
        raise CustomException(e, sys)