import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, UpSampling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
# ADD THIS LINE BELOW:
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
BATCH_SIZE = 256          # Increased for contrastive efficiency
LATENT_DIM = 392           # Adjust to match your actual flattening size
CONTRASTIVE_WEIGHT = 0.5   # How much weight to give separation force

### --- LOSS FUNCTIONS ---
def custom_weighted_binary_crossentropy(zero_weight, one_weight):
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        bce_loss = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
        weights = tf.where(tf.equal(y_true, 1), one_weight, zero_weight)
        weighted_bce_loss = weights * bce_loss
        return -tf.reduce_mean(weighted_bce_loss)
    return loss

def weighted_binary_crossentropy(class_weights):
    class_weights = tf.cast(class_weights, tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        bce_loss = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
        class_weights_tensor = tf.gather(class_weights, tf.cast(y_true, tf.int32))
        weighted_bce_loss = class_weights_tensor * bce_loss
        return -tf.reduce_mean(weighted_bce_loss)
    return loss

### --- DATA LOADING (Hybrid MNIST/Fashion) ---

print("Loading Datasets...")

# 1. LOAD BOTH
(mnist_x_train, mnist_y_train), _ = tf.keras.datasets.mnist.load_data()
(fashion_x_train, fashion_y_train), _ = tf.keras.datasets.fashion_mnist.load_data()

# 2. SLICE 50/50
mnist_half = mnist_x_train[:len(mnist_x_train)//2]
fashion_half = fashion_x_train[:len(fashion_x_train)//2]

X_full = np.concatenate((mnist_half, fashion_half), axis=0)

# Normalize / Reshape
X = X_full.reshape(X_full.shape[0], 28, 28, 1).astype('float32') / 255.0
Y = np.zeros((X.shape[0], 3)) # Dummy targets

print(f"Hybrid Dataset Size: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# ---------------------------------------------------------
# FIX: ADDING THE REQUIRED 'call' METHOD
# ---------------------------------------------------------

LATENT_SPATIAL_SIZE = 7  # Based on 2x MaxPool on 28x28 input
NUM_FILTERS_IN_LATENT = 8 # From your Encoder def

class ContrastiveAutoEncoder(Model):
    """
    AutoEncoder with Contrastive Loss in Latent Space
    """
    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.temperature = 0.3

    # THIS IS THE MISSING METHOD
    def call(self, inputs):
        """
        Standard Forward Pass.
        Input Image -> Encode(Z) -> Decode(Reconstruction)
        This tells Keras what happens during .fit(), .predict(), or .evaluate()
        """
        z_vector = self.encode(inputs)     # Flatten to 2D
        reconstruction = self.decode(z_vector) # Reshape back to 4D
        return reconstruction

    def encode(self, x):
        # Run through convolutions
        z_conv = self.encoder(x)
        # Flatten to 2D Vector
        return tf.reshape(z_conv, [tf.shape(x)[0], -1])

    def decode(self, z_vector):
        # Reshape 2D Vector back into 4D Image Tensor
        z_image_like = tf.reshape(
            z_vector, 
            [tf.shape(z_vector)[0], LATENT_SPATIAL_SIZE, LATENT_SPATIAL_SIZE, NUM_FILTERS_IN_LATENT]
        )
        # Pass through upsampling/convolutional stack
        return self.decoder(z_image_like)

    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            
            # --- 1. RECONSTRUCTION BRANCH ---
            z = self.encode(x)
            reconstruction = self.decode(z)
            
            recon_loss = tf.keras.losses.binary_crossentropy(
                x, reconstruction
            )
            recon_loss = tf.reduce_mean(recon_loss)
            
            # --- 2. CONTRASTIVE BRANCH ---
            # We augment the INPUT images slightly to create "Positive Pairs"
            augmented_x = tf.image.random_flip_left_right(
                tf.image.random_brightness(x, max_delta=0.2)
            )
            z_aug = self.encode(augmented_x)
            
            # Normalize for Cosine Similarity
            z_norm = tf.linalg.l2_normalize(z, axis=1)
            z_aug_norm = tf.linalg.l2_normalize(z_aug, axis=1)
            
            # Similarity Matrix (Batch vs Batch)
            sim_matrix = tf.matmul(z_norm, z_aug_norm, transpose_b=True)
            
            # Scale Temperature
            sim_matrix /= self.temperature
            
            # Labels: Diagonal represents Positive Pair (Image A vs Augmented Image A)
            labels_contrastive = tf.range(tf.shape(x)[0])
            
            # Compute NT-Xent Style Cross Entropy
            contrastive_loss = tf.keras.losses.sparse_categorical_crossentropy(
                labels_contrastive, sim_matrix, from_logits=True
            )
            contrastive_loss = tf.reduce_mean(contrastive_loss)

            # Total Weighted Loss
            total_loss = recon_loss + (0.5 * contrastive_loss)
        
        # Apply gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "contrastive_loss": contrastive_loss
        }


# Build Layers
encoder = Sequential()
encoder.add(Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
encoder.add(layers.Dropout(0.2))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same')) 
encoder.add(layers.Dropout(0.2))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(layers.Flatten())
encoder.add(layers.Dense(LATENT_DIM, activation='linear'))
# Projecting to the exact dimensionality expected by the Target Network


decoder = Sequential()
decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))  
decoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# Instantiate Hybrid Model
hybrid_ae = ContrastiveAutoEncoder(encoder, decoder, LATENT_DIM)

hybrid_ae.compile(optimizer=Adam(learning_rate=0.001))

print("\nStarting Training with Auxiliary Contrastive Loss...")
# Train using the custom logic
history = hybrid_ae.fit(
    x=X_train,
    y=y_train,   # Make sure this matches the lowercase 'y' from train_test_split
    epochs=30,
    batch_size=BATCH_SIZE
)

# ---------------------------------------------------------
# EVALUATION & EXPORT
# ---------------------------------------------------------
print("_______________________________________________________________________")
print("Evaluation")
print("_______________________________________________________________________")

# Verify Reconstruct
reconstructed_test = hybrid_ae.predict(X_test)
threshold = 0.5
test_acc = np.mean(np.equal(reconstructed_test >= threshold, X_test >= threshold))
print(f"Reconstruction Accuracy: {test_acc:.4f}")

# Export
tf.saved_model.save(hybrid_ae, "saved_cnnae_model_dir")
tf.saved_model.save(hybrid_ae.encoder, "saved_cnne_model_dir")