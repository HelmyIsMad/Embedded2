import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import librosa

model = models.load_model('speaker_id_model.h5')

sample = open('test.wav', 'rb').read()
y, sr = librosa.load(sample)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=320)
mfcc = np.expand_dims(mfcc, axis=-1)
mfcc = np.expand_dims(mfcc, axis=0)
print(model.predict(mfcc))