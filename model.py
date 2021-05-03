import time, os,re, csv, sys, uuid, joblib
import pickle
from datetime import date
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report

## import logging functions once complete

## model specific variables (iterate the version and note with each change)
if not os.path.exists(os.path.join(".", 'models')):
    os.mkdir('models')

MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = 'SVM on AAVAIL churn'
SAVED_MODEL  =os.path.join("models", 'model-{}.joblib'.format(re.sub("\.", "_", str(MODEL_VERSION))))

def load_data(filename):
    if not os.path.exists(os.path.join(".", "data")):
        os.mkdir("data")

    data_dir = os.path.join(".", "data")
    df = pd.read_csv(os.path.join(data_dir,filename))

    ## pull out the target and remove uneeded columns
    _y = df.pop('is_subscriber')
    y = np.zeros(_y.size)
    y[_y ==0] = 1

    df.drop(columns = ['customer_id','customer_name'], inplace = True)
    X = df

    return (X,y)








def get_preprocessor():
    pass

def model_train(test = False):
    pass

def model_predict():
    pass

def model_load():
    pass



