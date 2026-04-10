import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import os

# Suppress TensorFlow warnings
# silence_tensorflow()

# ==========================================
# 1. LOAD THE SAVED DATASET
# ==========================================
print("Loading dataset...")
if not os.path.exists('speaker_dataset.npz'):
    raise FileNotFoundError("Could not find 'speaker_dataset.npz'. Run your extraction script first!")

data = np.load('speaker_dataset.npz')
x_train = data['x_train'].astype(np.float32) # Ensure float32 for TensorFlow
y_train = data['y_train']

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Configuration variables
NUM_FRAMES = 50
NUM_MFCC = 13
NUM_CLASSES = 6  # 5 speakers + 1 unknown

# ==========================================
# 2. BUILD THE NEURAL NETWORK
# ==========================================
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        
        # First Convolutional Block
        layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Classify
        layers.Flatten(),
        layers.Dropout(0.3), # Helps prevent overfitting on small datasets
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model((NUM_FRAMES, NUM_MFCC, 1), NUM_CLASSES)
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# ==========================================
# 3. TRAIN THE MODEL
# ==========================================
print("\nStarting Training...")
# shuffle=True ensures the AI sees a good mix of speakers in every training batch
history = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.2, shuffle=True)
model.save('speaker_id_model.h5')
exit()

# ==========================================
# 4. TFLITE INT8 QUANTIZATION (FOR STM32)
# ==========================================
print("\nQuantizing model to INT8...")

# The converter needs a sample of the data to figure out how to scale floats to int8
def representative_dataset():
    # Provide ~100 random samples from the training data
    dataset_size = x_train.shape[0]
    indices = np.random.choice(dataset_size, size=100, replace=False)
    for i in indices:
        # Yield one sample at a time, keeping the batch dimension: shape (1, 50, 13, 1)
        yield [x_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Enforce FULL integer quantization (crucial for microcontrollers without massive FPU RAM)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

# Save the raw .tflite file (useful if you want to inspect it in tools like Netron)
with open('speaker_id_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)
print("Saved: speaker_id_model.tflite")

# ==========================================
# 5. CONVERT TO C HEADER FILE
# ==========================================
print("\nConverting to C header file for STM32CubeIDE...")

def convert_to_c_array(bytes_data, file_name):
    # Convert bytes to hex format
    hex_array = [hex(b) for b in bytes_data]
    
    # Format as a C-style array
    c_str = f"// Automatically generated TensorFlow Lite Micro model\n"
    c_str += f"#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n"
    c_str += f"// Align to 4 bytes for 32-bit ARM Cortex-M architecture\n"
    c_str += f"const unsigned char g_model[] __attribute__((aligned(4))) = {{\n"
    
    # Add 12 items per line for readability
    for i in range(0, len(hex_array), 12):
        c_str += "    " + ", ".join(hex_array[i:i+12]) + ",\n"
        
    c_str += f"}};\n\nconst int g_model_len = {len(hex_array)};\n\n"
    c_str += f"#endif // MODEL_DATA_H\n"
    
    with open(file_name, "w") as f:
        f.write(c_str)

if __name__ == "__main__":
    convert_to_c_array(tflite_quant_model, "model_data.h")
    print("Saved: model_data.h")
    print("\nDONE! You can now copy 'model_data.h' to your STM32 project.")