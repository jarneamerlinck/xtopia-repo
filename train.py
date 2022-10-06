# Load packages
## Data wrangling
import pandas as pd
import numpy as np

## Model training


## Model saving
import pickle

## Parser arguments
import argparse


# Set our parser arguments. 
parser = argparse.ArgumentParser(
    description='Painter image prediction')

parser.add_argument('--mlflow_run', default=0, type=int,
                    help="Running as an experiment or not. Don't change this, this gets automatically changed by the MLFlow default parameters")

args = parser.parse_args()

if args.mlflow_run:
    from mlflow import log_metric, log_param, log_artifacts

# Main function for training model on split train data and evaluating on validation data
def main():
    # Read preprocessed training data
  
    # Seperate our features from our outcome variable
    
    # Split our non-test set into a training and validation set
    
    # Train model

    # Make prediction on our validation dataset
    
    # Show our model accuracy on the validation data

    # Log hyperparameters of the model and output metrics
    if args.mlflow_run:
        print('Tracking MLFlow params & metrics')
        log_param('n_estimators',n_estimators)
        log_metric('Validation accuracy', accuracy)
        log_metric('Training accuracy', metrics.accuracy_score(y_train, model.predict(X_train))) # This can be interesting to see how much we're overfitting on our data
    
    # Train model on the full dataset
    
    # Save our model weights and parameters to a pickle file
    model_name = 'models/trained_model.pkl'
    with open(model_name, 'wb') as file:
        pickle.dump(model, file)
    
if __name__=='__main__':
    main()