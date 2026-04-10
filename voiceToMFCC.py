import os
import librosa
import numpy as np

# Configuration
DATASET_PATH = "TrainingVoiceRecords/" # Path to your folders
SAMPLE_RATE = 16000
DURATION = 1.0       # 1 second chunks
N_MFCC = 13
HOP_LENGTH = 320     # 16000 / 320 = 50 frames per second
N_FFT = 512          # CMSIS-DSP uses 512-point FFT by default

X_data = []
y_labels = []

# Map folders to integer labels
# Make sure your folder names exactly match these keys
class_mapping = {
    "Speaker0": 0, "Speaker1": 1, "Speaker2": 2, 
    "Speaker3": 3, "Speaker4": 4, 
    "Unknown": 5
}

def apply_compression(audio):
    # Simple software AGC/Compression
    # This boosts lower signals more than higher ones
    return np.sign(audio) * np.power(np.abs(audio), 0.5)

def extract_mfcc_from_audio(file_path, label):
    # Load the audio file
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)

    # Adding noise
    noise_amplitude = 0.005 * np.random.uniform(0, 1) # Random noise level
    noise = np.random.randn(len(audio))
    audio += (noise_amplitude * noise)

    audio = librosa.effects.preemphasis(audio)
    audio = apply_compression(audio)
    
    # Calculate how many 1-second chunks we can get
    samples_per_chunk = int(SAMPLE_RATE * DURATION)
    num_chunks = len(audio) // samples_per_chunk
    
    for i in range(num_chunks):
        # Slice the audio into 1-second segments
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        chunk = audio[start:end]
        
        # Extract MFCCs
        # Output shape of librosa mfcc is (n_mfcc, time_frames) -> (13, 50)
        mfcc = librosa.feature.mfcc(y=chunk, sr=SAMPLE_RATE, n_mfcc=N_MFCC, 
                                    n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # 1. Subtract the mean (removes the microphone EQ curve)
        mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
        mfcc_centered = mfcc - mfcc_mean

        # 2. Divide by standard deviation (removes volume/gain differences)
        mfcc_std = np.std(mfcc_centered, axis=1, keepdims=True)
        mfcc_normalized = mfcc_centered / (mfcc_std + 1e-8) # 1e-8 prevents division by zero

        # 3. Transpose for Neural Network
        mfcc_final = mfcc_normalized.T # Shape becomes (50, 13)
            
        X_data.append(mfcc_normalized)
        y_labels.append(label)

print("Processing audio files... This might take a minute.")

for folder_name in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder_name)
    if os.path.isdir(folder_path) and folder_name in class_mapping:
        label = class_mapping[folder_name]
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                extract_mfcc_from_audio(file_path, label)
                print(f"Processed {file_name} -> Class {label}")

# Convert lists to NumPy arrays and format for Keras
x_train = np.array(X_data)[..., np.newaxis] # (Samples, 50, 13, 1)
y_train = np.array(y_labels)

print(f"\nExtraction Complete!")
print(f"Total samples extracted: {len(x_train)}")

# Save the dataset to a NumPy file
# This creates a single compressed file containing both arrays
np.savez_compressed('speaker_dataset.npz', x_train=x_train, y_train=y_train)

print("Dataset successfully saved to 'speaker_dataset.npz'")