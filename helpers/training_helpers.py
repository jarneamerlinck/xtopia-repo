import numpy as np
import pandas as pd
import librosa
import os
import sys
import re

import mlflow
from mlflow import log_metric, log_param, log_artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.keras import log_model

# Global vars

FEATURES_PATH = "data/output/features.csv"


def load_CREMA(path:str) -> pd.core.frame.DataFrame:
    crema_directory_list = os.listdir(path)

    file_emotion = []
    file_path = []

    for file in crema_directory_list:
        file_path.append(path + file)
        
        part=file.split('_')
        if part[0] == "AudioWAV":
            pass #print(part) #folder
        elif part[2] == 'SAD':
            file_emotion.append('sad')
        elif part[2] == 'ANG':
            file_emotion.append('angry')
        elif part[2] == 'DIS':
            file_emotion.append('disgust')
        elif part[2] == 'FEA':
            file_emotion.append('fear')
        elif part[2] == 'HAP':
            file_emotion.append('happy')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')
            
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    path_df = pd.DataFrame(file_path, columns=['Path'])
    Crema_df = pd.concat([emotion_df, path_df], axis=1)
    return Crema_df

def load_RAVDESS(path:str) -> pd.core.frame.DataFrame:
    ravdess_directory_list = os.listdir(path)

    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        actor = os.listdir(path + dir)
        
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            try:
                file_emotion.append(int(part[2]))
                file_path.append(path + dir + '/' + file)
            except:
                pass
                #print(part) # Folders themself
            
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
    return Ravdess_df

def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2, offset=0.6, sr=8025)
    
    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def save_model(model, name_of_model, file_name, dir="models"):
    checkpoint_path = f"{dir}/{name_of_model}/{file_name}.ckpt"
    checkpoint_dir = f"{dir}/{name_of_model}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model.save(checkpoint_path)
    
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

def train_with_mlflow(run_name, model, compile_kwargs, fit_kwargs, optional_params={}):
  ''' 
  <Parameters>
  model: Keras model
  compile_kwargs: dict training algo settings (e.g.optimizer, loss, metrics)
  fit_kwargs: dict training data and hyper-parameters (e.g：trainin set, validation_split, epoch, batch_size, verbosity)
  '''
   
  # use Context Manager to encapsulate an MLFlow run
  with mlflow.start_run(run_name=run_name) as run:
     
    ##(1) initialize model
    model = model()
       
    # supply training algo settings
    model.compile(**compile_kwargs)
     
    # fit model, save training outputs to the "history" variable for tracking
    history = model.fit(**fit_kwargs)
     
    ##(2) MLFlow tracking
    for param_key, param_value in {**compile_kwargs, **fit_kwargs, **optional_params}.items():
      if param_key not in ['x', 'y']:
         
        # use log_param() to track hyper-parameters (except training dataset x,y)
        mlflow.log_param(param_key, param_value)
     
    for key, values in history.history.items():
      for i, v in enumerate(values):
         
        # use log_metric() to track evaluation metrics
        mlflow.log_metric(key, v, step=i)
 
    for i, layer in enumerate(model.layers):
       
      # use log_param() to track model.layer (details of each CNN layer)
      mlflow.log_param(f'hidden_layer_{i}_units', layer.output_shape)
         
    # use log_model() to track output Keras model (accessible through the MLFlow UI)
    log_model(model, 'keras_model')
   
     
    ##(3) sketch loss
    fig = view_loss(history)
     
    # save loss picture, use log_artifact() to track it in MLFLow UI
    fig.savefig('train-validation-loss.png')
    mlflow.log_artifact('train-validation-loss.png')
     
    # return MLFLow run context
    # this run variable contains the runID and experimentID that is essential to
    # retrieving our training outcomes programatically
    return run

def model_summary_to_dataframe(model, args):
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
    