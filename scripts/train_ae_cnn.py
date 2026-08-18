import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, UpSampling2D, Input, Reshape
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001

# ARCHITECTURE MATH
# Encoder reduces 28x28 -> 7x7 via pooling.
# We want Latent Vector = 128.
# Decoder must project 128 -> Volume(7x7x8) = 392 to restart Conv layers.
LATENT_SPATIAL_SIZE = 7 
NUM_FILTERS_IN_LATENT = 8
LATENT_DIM = 128  # User Requirement

CONTRASTIVE_WEIGHT = 0.5
TEMPERATURE = 0.3

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
print("Loading Datasets...")
(mnist_x_train, _), _ = tf.keras.datasets.mnist.load_data()
(fashion_x_train, _), _ = tf.keras.datasets.fashion_mnist.load_data()

# 50/50 Hybrid
mnist_half = mnist_x_train[:len(mnist_x_train)//2]
fashion_half = fashion_x_train[:len(fashion_x_train)//2]
X_full = np.concatenate((mnist_half, fashion_half), axis=0)

# Normalize
X = X_full.reshape(X_full.shape[0], 28, 28, 1).astype('float32') / 255.0
Y_dummy = np.zeros((X.shape[0], 1)) 

X_train, X_test, y_train, y_test = train_test_split(X, Y_dummy, test_size=0.2)
print(f"Hybrid Dataset Size: {X_train.shape}")

# ---------------------------------------------------------
# MODEL DEFINITION
# ---------------------------------------------------------

class ContrastiveAutoEncoder(Model):
    def __init__(self, encoder, decoder, latent_dim, temperature=0.3, contrastive_weight=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        
    def call(self, inputs):
        z_vector = self.encode(inputs)
        reconstruction = self.decode(z_vector)
        return reconstruction

    def encode(self, x):
        z_conv = self.encoder(x)
        # Ensure output is (Batch, 128)
        return tf.reshape(z_conv, [tf.shape(x)[0], -1])

    def decode(self, z_vector):
        # Decoder now handles the Dense Projection + Reshape internally
        return self.decoder(z_vector)

    def train_step(self, data):
        if isinstance(data, tuple):
            x, _ = data
        else:
            x = data
            
        with tf.GradientTape() as tape:
            # --- 1. RECONSTRUCTION BRANCH ---
            z = self.encode(x)
            reconstruction = self.decode(z)
            
            recon_loss = tf.keras.losses.binary_crossentropy(x, reconstruction)
            recon_loss = tf.reduce_mean(recon_loss)
            
            # --- 2. CONTRASTIVE BRANCH ---
            augmented_x = tf.image.random_flip_left_right(x)
            augmented_x = tf.image.adjust_brightness(augmented_x, tf.random.uniform([], -0.2, 0.2))
            augmented_x = tf.clip_by_value(augmented_x, 0.0, 1.0)
            
            z_aug = self.encode(augmented_x)
            
            z_norm = tf.linalg.l2_normalize(z, axis=1)
            z_aug_norm = tf.linalg.l2_normalize(z_aug, axis=1)
            
            sim_matrix = tf.matmul(z_norm, z_aug_norm, transpose_b=True)
            sim_matrix /= self.temperature
            
            labels_contrastive = tf.range(tf.shape(x)[0])
            
            contrastive_loss = tf.keras.losses.sparse_categorical_crossentropy(
                labels_contrastive, sim_matrix, from_logits=True
            )
            contrastive_loss = tf.reduce_mean(contrastive_loss)

            total_loss = recon_loss + (self.contrastive_weight * contrastive_loss)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Gradient Clipping for Stability
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "contrastive_loss": contrastive_loss
        }

# ---------------------------------------------------------
# BUILD NETWORKS
# ---------------------------------------------------------

# Encoder: 28x28 -> 128 Vector
encoder = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.Dropout(0.2),
    MaxPooling2D((2, 2), padding='same'), # 28 -> 14
    Conv2D(64, (3, 3), activation='relu', padding='same'), 
    layers.Dropout(0.2),
    MaxPooling2D((2, 2), padding='same'), # 14 -> 7
    Flatten(),                            # 7*7*64 = 3136
    Dense(LATENT_DIM, activation='linear') # 3136 -> 128
])

# Decoder: 128 Vector -> 28x28 Image
# Must Project 128 -> 392 (7*7*8) before Reshaping
decoder_volume = LATENT_SPATIAL_SIZE * LATENT_SPATIAL_SIZE * NUM_FILTERS_IN_LATENT

decoder = Sequential([
    Input(shape=(LATENT_DIM,)),
    Dense(decoder_volume, activation='relu'),          # 128 -> 392
    Reshape((LATENT_SPATIAL_SIZE, LATENT_SPATIAL_SIZE, NUM_FILTERS_IN_LATENT)), # 392 -> 7x7x8
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),                              # 7 -> 14
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),                              # 14 -> 28
    Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

# Instantiate
hybrid_ae = ContrastiveAutoEncoder(
    encoder, 
    decoder, 
    latent_dim=LATENT_DIM,
    temperature=TEMPERATURE,
    contrastive_weight=CONTRASTIVE_WEIGHT
)

hybrid_ae.compile(optimizer=Adam(learning_rate=LEARNING_RATE))

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
print("\nStarting Training...")
print(f"Latent Dim: {LATENT_DIM}")
print(f"Decoder Projection: {LATENT_DIM} -> {decoder_volume} (Reshaped to 7x7x{NUM_FILTERS_IN_LATENT})")

history = hybrid_ae.fit(
    x=X_train,
    y=y_train, 
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------
print("Saving Models...")
tf.saved_model.save(hybrid_ae, "saved_cnnae_model_dir")
tf.saved_model.save(hybrid_ae.encoder, "saved_cnne_model_dir")
print("Done.")