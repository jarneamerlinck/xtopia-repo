import numpy as np
import pandas as pd
import os
import sys
import re
import pickle

import mlflow
from mlflow import log_metric, log_param, log_artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.keras import log_model
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Global vars
FEATURES_PATH = "data/output/features.csv"
LABELS_PATH = "data/output/labels.csv"
ENCODER_PICKLE = "data/backup/encoder.pkl"

def save_model(model, name_of_model, file_name, dir="models"):
    checkpoint_path = f"{dir}/{name_of_model}/{file_name}.ckpt"
    checkpoint_dir = f"{dir}/{name_of_model}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model.save(checkpoint_path)
    
def load_model(name_of_model, file_name, dir="models"):
    checkpoint_path = f"{dir}/{name_of_model}/{file_name}.ckpt"
    model = keras.models.load_model(checkpoint_path)
    return model
   
def classification_report_to_dataframe(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        if line =='':
            pass
        elif 'accuracy' in line:
            row_data = re.split(' +', line)
            accuracy = row_data[2]
            break
        else:
            row_data = re.split(' +', line)
            row['class'] = row_data[1]
            row['precision'] = float(row_data[2])
            row['recall'] = float(row_data[3])
            row['f1_score'] = float(row_data[4])
            row['support'] = float(row_data[5])
            report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe, accuracy

def model_summary_to_MLFlow(model, model_name, model_version, args):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    summ_string = "\n".join(stringlist)
    print(summ_string) # entire summary in a variable

    table = stringlist[1:-4][1::2] # take every other element and remove appendix

    new_table = []
    for entry in table:
        entry = re.split(r'\s{2,}', entry)[:-1] # remove whitespace
        new_table.append(entry)

    df = pd.DataFrame(new_table[1:], columns=new_table[0])
    if args.mlflow_run:
        log_param(f"_number_of_hidden_layers",df.shape[0])
        log_param(f"_model_name",model_name)
        log_param(f"_model_version",model_version)
    
    number_of_current_layer = 0
    columns = df.columns
    for r in range(df.shape[0]):
        f = df.iloc[r,0]
        output_shape = df.iloc[r,1]
        params = df.iloc[r,2]
        print(f, output_shape, params)
        if args.mlflow_run:
            log_param(f"hidden_layer_{number_of_current_layer:03}_model",f)
            log_param(f"hidden_layer_{number_of_current_layer:03}_shape",output_shape)
            log_param(f"hidden_layer_{number_of_current_layer:03}_numer_of_params",params)
            
            number_of_current_layer+=1

def save_encoder(encoder: OneHotEncoder):
    with open(ENCODER_PICKLE, 'wb') as file:
        pickle.dump(encoder, file)

def load_encoder() -> OneHotEncoder:
    pickle_off = open (ENCODER_PICKLE, "rb")
    return pickle.load(pickle_off)