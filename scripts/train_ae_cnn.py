#Contrastive Autoencoder with Dual View Augmentation
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, UpSampling2D, Input, Reshape, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
BATCH_SIZE = 256
EPOCHS = 20  # Increased with EarlyStopping
LEARNING_RATE = 0.003

# ARCHITECTURE MATH
LATENT_SPATIAL_SIZE = 7 
NUM_FILTERS_IN_LATENT = 8
LATENT_DIM = 128 

# Regularization and Temperature Config
L2_REG_FACTOR = 1e-4
TEMP_INIT = 1.0
TEMP_MIN = 0.5
TEMP_MAX = 2.0
CONTRASTIVE_WEIGHT = 0.5

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
print("Loading Datasets...")
(mnist_x_train, _), _ = tf.keras.datasets.mnist.load_data()
(fashion_x_train, _), _ = tf.keras.datasets.fashion_mnist.load_data()

# 50/50 Hybrid
mnist_half = mnist_x_train[:len(mnist_x_train)//2]
fashion_half = fashion_x_train[:len(fashion_x_train)//2]
X_full = fashion_x_train#Only Fashion-MNIST for now#np.concatenate((mnist_half, fashion_half), axis=0)

# Normalize
X = X_full.reshape(X_full.shape[0], 28, 28, 1).astype('float32') / 255.0
Y_dummy = np.zeros((X.shape[0], 1)) 

X_train, X_test, y_train, y_test = train_test_split(X, Y_dummy, test_size=0.2)
print(f"Hybrid Dataset Size: {X_train.shape}")

# ---------------------------------------------------------
# DATA AUGMENTATION PIPELINE
# ---------------------------------------------------------
print("Setting up Data Augmentation Pipeline...")

# Define augmentation layer
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.01, 0.01),
    layers.RandomZoom(0.01),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# Function to generate TWO augmented views for contrastive learning
def augment_contrastive(x, y):
    x1 = data_augmentation(x, training=True)
    x2 = data_augmentation(x, training=True)
    return (x1, x2), y

# Create tf.data Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Apply augmentation ONLY to training data
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.map(augment_contrastive, num_parallel_calls=tf.data.AUTOTUNE)

# Validation data: NO augmentation
val_dataset = val_dataset.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch for performance
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------
# MODEL DEFINITION
# ---------------------------------------------------------

class ContrastiveAutoEncoder(Model):
    def __init__(self, encoder, decoder, latent_dim, temp_init=1.0, temp_min=0.5, temp_max=2.0, contrastive_weight=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.contrastive_weight = contrastive_weight
        self.temp_min = temp_min
        self.temp_max = temp_max
        
        # Learnable Temperature Variable
        self.temperature = self.add_weight(
            name="learnable_temperature",
            shape=[],
            initializer=tf.keras.initializers.Constant(temp_init),
            trainable=True
        )
        
    def call(self, inputs, training=False):
        if isinstance(inputs, tuple):
            x = inputs[0]
        else:
            x = inputs
        z_vector = self.encode(x, training=training)
        reconstruction = self.decode(z_vector, training=training)
        return reconstruction

    def encode(self, x, training=False):
        z_conv = self.encoder(x, training=training)
        return tf.reshape(z_conv, [tf.shape(x)[0], -1])

    def decode(self, z_vector, training=False):
        return self.decoder(z_vector, training=training)

    def train_step(self, data):
        # Unpack the two augmented views from the pipeline
        if isinstance(data, tuple):
            (x1, x2), _ = data
        else:
            x1, x2 = data, data
            
        with tf.GradientTape() as tape:
            # --- 1. RECONSTRUCTION BRANCH (using x1) ---
            z = self.encode(x1, training=True)
            reconstruction = self.decode(z, training=True)
            
            recon_loss = tf.keras.losses.binary_crossentropy(x1, reconstruction)
            recon_loss = tf.reduce_mean(recon_loss)
            
            # --- 2. CONTRASTIVE BRANCH (x1 vs x2) ---
            z1 = self.encode(x1, training=True)
            z2 = self.encode(x2, training=True)
            
            z1_norm = tf.linalg.l2_normalize(z1, axis=1)
            z2_norm = tf.linalg.l2_normalize(z2, axis=1)
            
            t_clipped = tf.clip_by_value(self.temperature, self.temp_min, self.temp_max)
            
            sim_matrix = tf.matmul(z1_norm, z2_norm, transpose_b=True)
            sim_matrix /= t_clipped
            
            labels_contrastive = tf.range(tf.shape(x1)[0])
            
            contrastive_loss = tf.keras.losses.sparse_categorical_crossentropy(
                labels_contrastive, sim_matrix, from_logits=True
            )
            contrastive_loss = tf.reduce_mean(contrastive_loss)

            total_loss = recon_loss + (self.contrastive_weight * contrastive_loss)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Enforce Temperature Constraints
        self.temperature.assign(tf.clip_by_value(self.temperature, self.temp_min, self.temp_max))
        
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "contrastive_loss": contrastive_loss,
            "temperature": self.temperature
        }

    def test_step(self, data):
        # --- VALIDATION LOGIC ---
        if isinstance(data, tuple):
            x, _ = data
        else:
            x = data
            
        # Inference mode (training=False)
        z = self.encode(x, training=False)
        reconstruction = self.decode(z, training=False)
        
        # Calculate ONLY Reconstruction Loss for validation
        recon_loss = tf.keras.losses.binary_crossentropy(x, reconstruction)
        recon_loss = tf.reduce_mean(recon_loss)
        
        return {
            "loss": recon_loss,
            "reconstruction_loss": recon_loss
        }

# ---------------------------------------------------------
# BUILD NETWORKS
# ---------------------------------------------------------

l2_reg = regularizers.l2(L2_REG_FACTOR)

# Encoder: 28x28 -> 128 Vector
encoder = Sequential([
    Input(shape=(28, 28, 1)),
    
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
    BatchNormalization(),
    Dropout(0.2),
    MaxPooling2D((2, 2), padding='same'),
    
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
    BatchNormalization(),
    Dropout(0.2),
    MaxPooling2D((2, 2), padding='same'),
    
    Flatten(),
    Dense(LATENT_DIM, activation='linear', kernel_regularizer=l2_reg)
])

# Decoder: 128 Vector -> 28x28 Image
decoder_volume = LATENT_SPATIAL_SIZE * LATENT_SPATIAL_SIZE * NUM_FILTERS_IN_LATENT

decoder = Sequential([
    Input(shape=(LATENT_DIM,)),
    
    Dense(decoder_volume, activation='relu', kernel_regularizer=l2_reg),
    BatchNormalization(),
    Reshape((LATENT_SPATIAL_SIZE, LATENT_SPATIAL_SIZE, NUM_FILTERS_IN_LATENT)),
    
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
    BatchNormalization(),
    UpSampling2D((2, 2)),
    
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2_reg),
    BatchNormalization(),
    UpSampling2D((2, 2)),
    
    Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

# Instantiate
hybrid_ae = ContrastiveAutoEncoder(
    encoder, 
    decoder, 
    latent_dim=LATENT_DIM,
    temp_init=TEMP_INIT,
    temp_min=TEMP_MIN,
    temp_max=TEMP_MAX,
    contrastive_weight=CONTRASTIVE_WEIGHT
)

hybrid_ae.compile(optimizer=Adam(learning_rate=LEARNING_RATE))

# ---------------------------------------------------------
# TRAINING WITH CALLBACKS
# ---------------------------------------------------------
print("\nStarting Training...")
print(f"Latent Dim: {LATENT_DIM}")
print(f"Learnable Temperature: Init={TEMP_INIT}, Range=[{TEMP_MIN}, {TEMP_MAX}]")
print(f"L2 Regularization: {L2_REG_FACTOR}")
print("Using tf.data pipeline with dual-view augmentation")

# Early Stopping: Monitor Validation Loss
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Reduce LR on Plateau
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.8,
    patience=1,
    min_lr=1e-6,
    verbose=1
)

history = hybrid_ae.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stop, reduce_lr]
)

# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------
print("\nSaving Models...")
tf.saved_model.save(hybrid_ae, "saved_cnnae_model_dir")
tf.saved_model.save(hybrid_ae.encoder, "saved_cnne_model_dir")
print("Done.")

# ---------------------------------------------------------
# VISUALIZATION (Optional)
# ---------------------------------------------------------
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['reconstruction_loss'], label='Train Recon')
plt.plot(history.history['val_reconstruction_loss'], label='Val Recon')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss')
plt.legend()
plt.title('Reconstruction Loss Over Epochs')

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history saved to 'training_history.png'")

# Generate sample reconstructions
print("\nGenerating sample reconstructions...")
sample_images = X_test[:10]
reconstructed = hybrid_ae.predict(sample_images, verbose=0)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(sample_images[i].squeeze(), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 10, i+11)
    plt.imshow(reconstructed[i].squeeze(), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.savefig('sample_reconstructions.png')
print("Sample reconstructions saved to 'sample_reconstructions.png'")