import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# ---------------------------------------------------------
# 1. PREPARE DATA (MNIST)
# ---------------------------------------------------------
print("Loading MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalization: MUST match what the Autoencoder saw during its own training!
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to add channel dimension (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Train Shape: {x_train.shape}")

# ---------------------------------------------------------
# 2. LOAD THE SAVED ENCODER & WRAP IT
# ---------------------------------------------------------
saved_encoder_path = "./saved_cnne_model_dir"
loaded_encoder_obj = tf.saved_model.load(saved_encoder_path)

class FrozenEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, saved_module, **kwargs):
        super(FrozenEncoderLayer, self).__init__(**kwargs)
        self.module = saved_module
        
    def call(self, inputs):
        return self.module(inputs)
    
    # IMPORTANT: Helps Keras understand how to save/load this custom layer
    def get_config(self):
        return super(FrozenEncoderLayer, self).get_config()

# Instantiate our wrapper
frozen_encoder_block = FrozenEncoderLayer(loaded_encoder_obj)
frozen_encoder_block.trainable = False

# ---------------------------------------------------------
# 3. BUILD THE CLASSIFICATION MODEL
# ---------------------------------------------------------
inputs = tf.keras.Input(shape=(28, 28, 1))

# Pass input through the FROZEN encoder
features = frozen_encoder_block(inputs)

# Flatten features for the Dense layers
x = layers.Flatten()(features)

# --- THE NEW MLP HEAD ---
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

# Create the final combined model
classifier = models.Model(inputs, outputs, name="MNIST_Classifier_From_Saved_Encoder")

# ---------------------------------------------------------
# 4. COMPILE & TRAIN
# ---------------------------------------------------------
classifier.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Integers 0-9
    metrics=['accuracy']
)

print("\nStarting Training on Frozen Encoder Features...")
history = classifier.fit(
    x_train, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

# ---------------------------------------------------------
# 5. EVALUATE
# ---------------------------------------------------------
loss, acc = classifier.evaluate(x_test, y_test)
print(f"\nFinal Test Accuracy with Frozen Encoder: {acc:.4f}")

# ---------------------------------------------------------
# 6. SAVE THE CLASSIFIER (NEW CODE)
# ---------------------------------------------------------

# Option A: Save as SavedModel (Directory) - Recommended for production/serving
# This saves the architecture, weights, and optimizer state
print("\nSaving model to 'saved_mnist_classifier_dir' ...")
classifier.save("saved_mnist_classifier_dir") 

# Option B: Save as HDF5 file (.h5) - Good for simple single-file storage
# print("Saving model to 'mnist_classifier.h5' ...")
# classifier.save("mnist_classifier.h5")

print("Model Saved Successfully.")