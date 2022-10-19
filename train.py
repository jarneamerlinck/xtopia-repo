# Load packages
## Data wrangling
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Activation
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
import re


from helpers.preprocessing_helpers import *
from helpers.training_helpers import *

## Parser arguments
import argparse


# Set our parser arguments. 
parser = argparse.ArgumentParser(
    description='xtopia voice emotion detection')

parser.add_argument('--mlflow_run', default=0, type=int,
                    help="Running as an experiment or not. Don't change this, this gets automatically changed by the MLFlow default parameters")

args = parser.parse_args()

if args.mlflow_run:
    from mlflow import log_metric, log_param, log_artifacts

def train_model(x_train, y_train):
    model_name = "NN_Sequential"
    model_version = "001"
           
    model = Sequential()
    model.add(Conv1D(256, 8, padding='same', input_shape=(x_train.shape[1],1))) 
    model.add(Activation('relu'))

    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))

    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))

    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(y_train.shape[1])) # Target class number
    model.add(Activation('softmax'))

    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    return model, model_name, model_version

# Main function for training model on split train data and evaluating on validation data
def main():
    # Read preprocessed training data
    data_path = pd.read_pickle(DATA_PATH_PICKLE)
    
    path = np.array(data_path.Path)[1]
    
    data, sample_rate = librosa.load(path)
    
    # Seperate our features from our outcome variable
    print("Start creating features")
    
    X, Y = [], []
    for path, emotion in zip(data_path.Path, data_path.Emotions):
        feature = get_features(path)
        for ele in feature:
            X.append(ele)
            Y.append(emotion)
            
    features = pd.DataFrame(X)
    features['labels'] = Y
    features.to_csv(FEATURES_PATH, index=False)
    features.head()
    # Split our non-test set into a training and validation set

    X = features.iloc[: ,:-1].values
    Y = features['labels'].values

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    print("Done creating features")
    
    # Train model
    model, model_name, model_version = train_model(x_train, y_train)
    
    print(model.summary())
    
    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
    history=model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[rlrp])
       
    pred_test = model.predict(x_test)
    y_pred = encoder.inverse_transform(pred_test)

    y_test = encoder.inverse_transform(y_test)
    
    # Show our model accuracy on the test data
    #accuracy = model.evaluate(x_test, y_test)
    report, accuracy = classification_report_to_dataframe(classification_report(y_test, y_pred))
    print(report)
    
    # Log output metrics
    if args.mlflow_run:
        print('Tracking MLFlow metrics')
        log_metric('Training accuracy', accuracy)
        #log_metric('Training accuracy', metrics.accuracy_score(y_train, model.predict(X_train))) # This can be interesting to see how much we're overfitting on our data
    
    
    # Save our model weights and parameters

    save_model(model, model_name, model_version)
    model_summary_to_MLFlow(model, model_name, model_version, args)
    save_encoder(encoder)

if __name__=='__main__':
    main()