import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import librosa
from scipy.io import wavfile
import os
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Script started", flush=True)

# Configuration (must match training)
SAMPLE_RATE = 16000
DURATION = 1.0       # 1 second
N_MFCC = 13
HOP_LENGTH = 320
N_FFT = 512

# Load the trained model
print("Loading model...", flush=True)
model = models.load_model('speaker_id_model.h5')
print("Model loaded successfully", flush=True)

# Load the test audio file using scipy
print("Loading audio file...", flush=True)
sr, y = wavfile.read('test.wav')
# Convert stereo to mono if needed
if len(y.shape) > 1:
    y = np.mean(y, axis=1)
# Resample if needed
if sr != SAMPLE_RATE:
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
else:
    y = y.astype(np.float32)
print(f"Audio loaded: sr={sr}->{SAMPLE_RATE}, duration={len(y)/SAMPLE_RATE:.2f}s", flush=True)

# Apply preprocessing (same as training)
def apply_compression(audio):
    return np.sign(audio) * np.power(np.abs(audio), 0.5)

y = librosa.effects.preemphasis(y)
y = apply_compression(y)
print(f"Preprocessing done: audio shape={y.shape}", flush=True)

# Extract 1-second chunks (same as training)
samples_per_chunk = int(SAMPLE_RATE * DURATION)

# Warm up librosa by calling it once
print("Pre-loading librosa...", flush=True)
_ = librosa.feature.mfcc(y=y[:samples_per_chunk], sr=SAMPLE_RATE, n_mfcc=N_MFCC, 
                         n_fft=N_FFT, hop_length=HOP_LENGTH)
print("Librosa ready", flush=True)

# Process each 1-second chunk
all_predictions = []
chunk_count = 0

for i in range(0, len(y) - samples_per_chunk + 1, samples_per_chunk):
    chunk = y[i:i+samples_per_chunk]
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=chunk, sr=SAMPLE_RATE, n_mfcc=N_MFCC, 
                                n_fft=N_FFT, hop_length=HOP_LENGTH)
    print(f"MFCC shape: {mfcc.shape}", flush=True)
    
    # Normalize (same as training)
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc_centered = mfcc - mfcc_mean
    mfcc_std = np.std(mfcc_centered, axis=1, keepdims=True)
    mfcc_normalized = mfcc_centered / (mfcc_std + 1e-8)
    
    # Ensure we have exactly 50 frames (trim or pad as needed)
    if mfcc_normalized.shape[1] > 50:
        mfcc_normalized = mfcc_normalized[:, :50]  # Trim to 50 frames
    elif mfcc_normalized.shape[1] < 50:
        pad_width = ((0, 0), (0, 50 - mfcc_normalized.shape[1]))
        mfcc_normalized = np.pad(mfcc_normalized, pad_width, mode='constant', constant_values=0)
    
    # Add channel dimension: (13, 50) -> (13, 50, 1) - same as training
    mfcc_input = np.expand_dims(mfcc_normalized, axis=-1)  # (13, 50, 1)
    mfcc_input = np.expand_dims(mfcc_input, axis=0)  # (1, 13, 50, 1)
    print(f"Input shape (after adjustment): {mfcc_input.shape}", flush=True)
    
    # Make prediction
    prediction = model(tf.constant(mfcc_input), training=False).numpy()
    print(f"Prediction: {prediction[0]}", flush=True)
    all_predictions.append(prediction[0])
    chunk_count += 1

# Map class indices to speaker labels
class_labels = ["Speaker0", "Speaker1", "Speaker2", "Speaker3", "Speaker4", "Unknown"]

if len(all_predictions) > 0:
    # Average predictions if multiple chunks
    avg_predictions = np.mean(all_predictions, axis=0)
    predicted_class = np.argmax(avg_predictions)
    confidence = avg_predictions[predicted_class]
    
    print(f"\nResults:", flush=True)
    print(f"Predictions (probabilities): {avg_predictions}", flush=True)
    print(f"Predicted Speaker: {class_labels[predicted_class]}", flush=True)
    print(f"Confidence: {confidence:.4f}", flush=True)
    print(f"Number of chunks processed: {chunk_count}", flush=True)
else:
    print("No audio chunks processed", flush=True)