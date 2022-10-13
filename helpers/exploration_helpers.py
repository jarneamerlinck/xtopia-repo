import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt


def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    return plt.show()

def create_spectrogram(data, sr, e):
    
    fig = plt.figure(figsize=(12, 3))
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')   
    return plt.colorbar()