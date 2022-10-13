# Load packages
## Data wrangling
import pandas as pd
import numpy as np

## Helpers
from helpers.preprocessing_helpers import *

# Main preprocessing function
def main():
    # Read data
    RAVDESS_PATH = "data/raw/RAVDESS/audio_speech_actors_01-24/"
    CREMA_PATH = "data/raw/CREMA_D/AudioWAV/"
    ravdess_df = load_RAVDESS(RAVDESS_PATH)
    crema_df = load_CREMA(CREMA_PATH)
    # Inspect our result

    # Save results to data/preprocessed
    data_path = pd.concat([ravdess_df], axis = 0)
    data_path.to_csv(DATA_PATH_FILE,index=False)
    data_path.to_pickle(DATA_PATH_PICKLE)
    
    
if __name__ == '__main__':
    main()