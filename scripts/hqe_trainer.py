# ---------------------------------------------------------
# HYPER QUERY ENCODER SCRIPT (MULTI-HOP + JOYSTICK TRICK + HYPERNETWORK PER HOP)
# ---------------------------------------------------------

# ---------------------------------------------------------
# SQLITE FIX: Must be at the TOP of the script
# ---------------------------------------------------------
import sys
try:
    # Try to load the modern binary we installed
    __import__('pysqlite3')
    # Replace the standard sqlite3 module with the new one
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If pysqlite3-binary is not installed, print an error
    print("ERROR: pysqlite3-binary not installed! Run 'pip install pysqlite3-binary'")
    exit()

# ---------------------------------------------------------
# NOW continue with normal imports...
# ---------------------------------------------------------

import os
import sys
import ast 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import chromadb


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CHROMA_DB_PATH = "./chroma_db_mnist_prototypes"
COLLECTION_NAME = "mnist_prototype_collection"

ENCODER_PATH = "./saved_cnne_model_dir"
VALUE_ENC_PATH = "./saved_mnist_classifier_dir" 

SAVE_PATH_HQE_SYSTEM = "./saved_hqe_hyper_multi_hop_system"
EMBEDDING_DIM = 128  # Must match your Frozen Encoder output size
NUM_NEIGHBORS = 10       
BATCH_SIZE = 128
EPOCHS = 25             
LEARNING_RATE = 0.001

# MULTI-HOP CONFIGURATION (From QUERY ENCODER SCRIPT)
NUM_HOPS = 4            # Number of stacked EQV blocks for multi-hop retrieval
                        # Each hop has its own CNN + Hypernetwork pair (1:1 ratio)

# PROTOTYPE CONFIGURATION (The "Joystick Trick")
NUM_VISUAL_CENTROIDS = 64   # K: Number of visual clusters
NUM_ACTIONS = 10            # N: Number of action classes (Fashion-MNIST labels)
NUM_PROTOTYPES = NUM_VISUAL_CENTROIDS * NUM_ACTIONS  # K × N total prototypes

# HYPERNETWORK CONFIGURATION (From HYPER SCRIPT)
TARGET_NET_ARCH = [64, 32]  # Hidden layers for generated target network
HYPER_INTERMEDIATE_DIM = 98  # Hypernetwork hidden dimension

# CONSOLIDATION FOR EDA (Optional: Set to True to save manifold snapshots)
ENABLE_CONSOLIDATION_EDA = False
EDA_SAVE_PATH = "./eda_manifold_snapshots"


# ---------------------------------------------------------
# HELPER: Robust SavedModel Caller
# ---------------------------------------------------------
def _extract_tensor(out):
    """SavedModels sometimes return dicts / tuples / single tensors."""
    if isinstance(out, dict):
        return list(out.values())[0]
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def call_frozen_encoder(module, x):
    """
    Robust caller for a tf.saved_model artifact.
    Tries: direct call -> serving signature -> first available signature.
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # 1) Direct call (this is what your original FrozenEncoderLayer used)
    for kwargs in ({}, {"training": False}):
        try:
            return _extract_tensor(module(x, **kwargs) if kwargs else module(x))
        except Exception:
            pass

    # 2) Signatures
    if hasattr(module, "signatures") and len(module.signatures) > 0:
        sig = module.signatures.get("serving_default", None)
        if sig is None:
            sig = list(module.signatures.values())[0]

        try:
            return _extract_tensor(sig(x))
        except Exception as e:
            raise RuntimeError(f"Failed to call encoder signature: {e}")

    raise RuntimeError("Could not call frozen encoder via direct call or signatures.")


def encode_images(module, images, batch_size=256):
    outs = []
    n = len(images)
    for i in range(0, n, batch_size):
        batch = images[i:i + batch_size]
        z = call_frozen_encoder(module, batch)
        z = tf.reshape(z, [tf.shape(z)[0], -1])
        outs.append(z.numpy())
    return np.concatenate(outs, axis=0)


# ---------------------------------------------------------
# 1. DATA PREPARATION (MNIST)
# ---------------------------------------------------------
print("_______________________________________________________________________")
print("Loading Data")
print("_______________________________________________________________________")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_full = np.concatenate((x_train, x_test), axis=0)
Y_full = np.concatenate((y_train, y_test), axis=0)

X_processed = X_full.reshape(X_full.shape[0], 28, 28, 1).astype('float32') / 255.0
Y_onehot = tf.keras.utils.to_categorical(Y_full, NUM_ACTIONS)

indices = np.arange(len(X_processed))
idx_train, idx_test, _, _ = train_test_split(indices, Y_full, test_size=0.2, stratify=Y_full)

X_tr = X_processed[idx_train]; y_tr_int = Y_full[idx_train]; y_tr_hot = Y_onehot[idx_train]
X_te = X_processed[idx_test]; y_te_int = Y_full[idx_test]; y_te_hot = Y_onehot[idx_test]


# ---------------------------------------------------------
# 2. GENERATE COMPOSITIONAL PROTOTYPE KEYS
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("Generating Compositional Prototype Keys (Visual Centroids + Action Offsets)")
print("_______________________________________________________________________")

# Load Frozen Encoder to generate visual centroids
print(f"Loading Frozen Encoder from {ENCODER_PATH}...")
loaded_encoder = tf.saved_model.load(ENCODER_PATH)
print("Frozen Encoder loaded successfully.")

# Generate latent vectors for all training data
print("Encoding training data for K-Means...")
Z_train = encode_images(loaded_encoder, X_tr, batch_size=256)
print(f"Z_train shape: {Z_train.shape}")

# K-Means to find Visual Centroids (The "Base of the Joystick")
print(f"Running K-Means with K={NUM_VISUAL_CENTROIDS}...")
kmeans = KMeans(n_clusters=NUM_VISUAL_CENTROIDS, random_state=42, n_init=10)
kmeans.fit(Z_train)
VISUAL_CENTROIDS = kmeans.cluster_centers_.astype('float32')
print(f"Generated {NUM_VISUAL_CENTROIDS} visual centroids.")

# Generate Fixed Action Offsets (The "Joystick Buttons" - Hyperspherical Distribution)
print(f"Generating {NUM_ACTIONS} fixed action offsets (Joystick Trick)...")

def generate_hyperspherical_offsets(dim, n_vectors):
    """
    Generates n_vectors evenly distributed on a hypersphere of given dimension.
    Uses repulsion-based optimization for even distribution.
    """
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dim).astype('float32')
    
    # Normalize each vector
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    # Apply repulsion to spread them out (simple iterative refinement)
    for _ in range(100):
        for i in range(n_vectors):
            repulsion = np.zeros(dim)
            for j in range(n_vectors):
                if i != j:
                    diff = vectors[i] - vectors[j]
                    dist = np.linalg.norm(diff)
                    if dist > 0.001:
                        repulsion += diff / (dist ** 2)
            vectors[i] += repulsion * 0.01
            vectors[i] /= np.linalg.norm(vectors[i])
    
    # Scale offsets to be a fraction of the centroid magnitude (e.g., 10%)
    centroid_magnitude = np.mean(np.linalg.norm(VISUAL_CENTROIDS, axis=1))
    offset_scale = centroid_magnitude * 0.1
    vectors = vectors * offset_scale
    
    return vectors

ACTION_OFFSETS = generate_hyperspherical_offsets(EMBEDDING_DIM, NUM_ACTIONS)
print(f"Action offsets generated with scale: {np.mean(np.linalg.norm(ACTION_OFFSETS, axis=1)):.4f}")

# Create Compositional Prototype Keys: K_centroids × N_actions
print(f"Creating {NUM_PROTOTYPES} compositional prototype keys...")
PROTOTYPE_KEYS = []
PROTOTYPE_LABELS = []
PROTOTYPE_OFFSETS = []

for k_idx in range(NUM_VISUAL_CENTROIDS):
    for a_idx in range(NUM_ACTIONS):
        prototype_key = VISUAL_CENTROIDS[k_idx] + ACTION_OFFSETS[a_idx]
        PROTOTYPE_KEYS.append(prototype_key)
        
        action_onehot = np.zeros(NUM_ACTIONS)
        action_onehot[a_idx] = 1.0
        PROTOTYPE_LABELS.append(action_onehot)
        PROTOTYPE_OFFSETS.append(ACTION_OFFSETS[a_idx])

PROTOTYPE_KEYS = np.array(PROTOTYPE_KEYS).astype('float32')
PROTOTYPE_LABELS = np.array(PROTOTYPE_LABELS).astype('float32')
PROTOTYPE_OFFSETS = np.array(PROTOTYPE_OFFSETS).astype('float32')

print(f"Prototype Keys shape: {PROTOTYPE_KEYS.shape}")
print(f"Prototype Labels shape: {PROTOTYPE_LABELS.shape}")
print(f"Prototype Offsets shape: {PROTOTYPE_OFFSETS.shape}")


# =============================================================================
# === SECTION 2.5: COMPUTE DIRECTIONAL VECTORS (JOYSTICK TRICK) ===
# =============================================================================
print("\n_______________________________________________________________________")
print("Computing Directional Vectors (Joystick Trick Pattern)")
print("_______________________________________________________________________")

# === THE KEY FIX: Prototype directions are JUST the normalized action offsets ===
DIRECTIONAL_PROTOTYPE_VECS = []

for k_idx in range(NUM_VISUAL_CENTROIDS):
    for a_idx in range(NUM_ACTIONS):
        # Direction is just the normalized offset (same for ALL centroids)
        anchor = np.zeros(EMBEDDING_DIM, dtype='float32')
        direction = ACTION_OFFSETS[a_idx] - anchor
        direction = tf.linalg.l2_normalize(direction, axis=0)
        DIRECTIONAL_PROTOTYPE_VECS.append(direction)

DIRECTIONAL_PROTOTYPE_VECS = np.array(DIRECTIONAL_PROTOTYPE_VECS).astype('float32')
print(f"Directional Prototype Vectors shape: {DIRECTIONAL_PROTOTYPE_VECS.shape}")
print(f"=== All {NUM_VISUAL_CENTROIDS} centroids share the same {NUM_ACTIONS} directions ===")

# Save to ChromaDB
print("\nSaving directional vectors to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

ids = [f"prototype_{k}_{a}" for k in range(NUM_VISUAL_CENTROIDS) for a in range(NUM_ACTIONS)]
metadatas = [
    {"visual_centroid": int(k), "action_id": int(a)} 
    for k in range(NUM_VISUAL_CENTROIDS) for a in range(NUM_ACTIONS)
]

try:
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])
except:
    pass

collection.add(
    embeddings=DIRECTIONAL_PROTOTYPE_VECS.tolist(),
    ids=ids,
    metadatas=metadatas
)
print(f"Saved {len(ids)} directional prototype vectors to ChromaDB.")

# Load as TensorFlow constants for training
MEM_BANK_VECS = tf.constant(DIRECTIONAL_PROTOTYPE_VECS)  # Global KNN directions
MEM_BANK_LABELS = tf.constant(PROTOTYPE_LABELS)
MEM_BANK_OFFSETS = tf.constant(PROTOTYPE_OFFSETS)
CENTROID_VECS = tf.constant(VISUAL_CENTROIDS)


# ---------------------------------------------------------
# 3. ARCHITECTURE DEFINITIONS (MULTI-HOP + JOYSTICK + HYPERNETWORK PER HOP)
# ---------------------------------------------------------

class FrozenEncoderLayer(layers.Layer):
    """Takes Image -> Flattened Latent Z from a tf.saved_model artifact."""
    def __init__(self, module, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self.trainable = False
        
    def call(self, inputs):
        res = call_frozen_encoder(self.module, inputs)
        return tf.reshape(res, [tf.shape(res)[0], -1]) 


class ResidualCNN(keras.Model):
    """
    Takes RAW IMAGE -> Outputs Adjustment Vector (Same Size as Z)
    Uses convolutions to extract local spatial patterns lost by the base AE.
    One instance per hop (1:1 with Hypernetwork).
    """
    def __init__(self, target_dim, hop_id=0):
        super().__init__()
        self.hop_id = hop_id
        
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name=f"hop{hop_id}_conv1", kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.drop1 = layers.Dropout(0.3)
        self.bn1 = layers.BatchNormalization(name=f"hop{hop_id}_bn1")
        self.pool1 = layers.MaxPooling2D((2, 2), name=f"hop{hop_id}_pool1")
        
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name=f"hop{hop_id}_conv2", kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.drop2 = layers.Dropout(0.3)
        self.bn2 = layers.BatchNormalization(name=f"hop{hop_id}_bn2")
        
        self.flatten = layers.Flatten(name=f"hop{hop_id}_flatten")
        self.dense_proj = layers.Dense(target_dim, activation='relu', name=f"hop{hop_id}_dense")
        self.out_layer = layers.Dense(target_dim, activation='linear', name=f"hop{hop_id}_out") 

    def call(self, raw_image_inputs):
        x = self.conv1(raw_image_inputs)
        x = self.drop1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.bn2(x)
        x = self.flatten(x)
        x = self.dense_proj(x)
        delta_z = self.out_layer(x)
        return delta_z


# =============================================================================
# === HYPERNETWORK: Generates Weights Based on Centroid Context (Per Hop) ===
# =============================================================================

def get_target_params_count(input_dim, arch_list, output_dim):
    """Calculate total parameters needed for the generated target network."""
    count = 0
    prev = input_dim
    for size in arch_list:
        count += prev * size + size  # weights + biases
        prev = size
    count += prev * output_dim + output_dim  # output layer
    return count

TOTAL_PARAMS_PER_HOP = get_target_params_count(EMBEDDING_DIM, TARGET_NET_ARCH, EMBEDDING_DIM)
print(f"\nEach Hypernetwork will generate {TOTAL_PARAMS_PER_HOP} parameters")
print(f"Total Hypernetwork Parameters: {TOTAL_PARAMS_PER_HOP * NUM_HOPS}")


class CentroidHypernetwork(keras.Model):
    """
    Generates network weights based on centroid context vector.
    One instance per hop (1:1 with ResidualCNN).
    """
    def __init__(self, output_param_count, hop_id=0):
        super().__init__()
        self.hop_id = hop_id
        self.intermediate_dim = HYPER_INTERMEDIATE_DIM
        self.net = keras.Sequential([
            layers.Dense(self.intermediate_dim, activation='relu', name=f"hyper_hop{hop_id}_dense1"),
            layers.Dense(output_param_count, activation='linear', name=f"hyper_hop{hop_id}_dense2")
        ])
    
    def call(self, centroid_context):
        return self.net(centroid_context)


class DynamicTargetNetwork(layers.Layer):
    """
    Executes an MLP defined purely by gen_params.
    Takes hop-refined query and applies hypernetwork-generated transformations.
    One instance per hop (1:1 with ResidualCNN and CentroidHypernetwork).
    """
    def __init__(self, arch_list, output_dim, hop_id=0):
        super().__init__()
        self.hop_id = hop_id
        self.arch_list = arch_list
        self.output_dim = output_dim
        
    def call(self, hop_query, generated_params):
        current_x = hop_query
        
        # Infer input dim dynamically
        current_dim = tf.shape(current_x)[-1] 
        offset = 0
        
        # Process Hidden Layers
        for i, next_dim in enumerate(self.arch_list):
            w_size = current_dim * next_dim
            b_size = next_dim
            
            w_flat = generated_params[:, offset : offset + w_size]
            b_val  = generated_params[:, offset + w_size : offset + w_size + b_size]
            
            offset += (w_size + b_size)
            
            w_matrix = tf.reshape(w_flat, [tf.shape(current_x)[0], current_dim, next_dim])
            out = tf.einsum('bi,bij->bj', current_x, w_matrix) + b_val
            out = tf.nn.relu(out)
            
            current_x = out
            current_dim = next_dim
            
        # Process Output Layer (Must map back to EMBEDDING_DIM)
        out_dim = self.output_dim
        w_size_out = current_dim * out_dim
        b_size_out = out_dim
        
        w_flat_out = generated_params[:, offset : offset + w_size_out]
        b_val_out  = generated_params[:, offset + w_size_out : offset + w_size_out + b_size_out]
        
        w_matrix_out = tf.reshape(w_flat_out, [tf.shape(current_x)[0], current_dim, out_dim])
        
        # Produces refined delta for this hop
        refined_delta = tf.einsum('bi,bij->bj', current_x, w_matrix_out) + b_val_out
        
        return refined_delta


# =============================================================================
# === MultiHopHyperRetriever (1:1 CNN + Hypernetwork Per Hop + Joystick) ===
# =============================================================================
class MultiHopHyperRetriever(Model):
    """
    Combines THREE mechanisms with 1:1 ratio per hop:
    1. Multi-Hop residual CNNs (from QUERY ENCODER) - one per hop
    2. Hypernetwork weight generation from centroids (from HYPER script) - one per hop
    3. Joystick Trick prototype attention (from QUERY ENCODER) - final retrieval
    
    Architecture per hop:
    [Raw Image] → [CNN Residual] → [Query + Delta] → [Normalize]
                                           ↓
                                    [Centroid Lookup]
                                           ↓
                                    [Hypernetwork] → [Generated Weights]
                                           ↓
                                    [Dynamic Target Net] → [Refined Delta]
                                           ↓
                                    [Query + Refined Delta] → [Next Hop]
    """
    def __init__(self, enc, num_hops, target_dim, hyper_arch, output_dim):
        super().__init__()
        self.enc = enc
        self.num_hops = num_hops
        self.target_dim = target_dim
        self.output_dim = output_dim
        
        # 1:1 Ratio: Each hop has its own CNN + Hypernetwork + Target Net
        self.hop_cnns = [ResidualCNN(target_dim=target_dim, hop_id=i) for i in range(num_hops)]
        self.hop_hypernets = [CentroidHypernetwork(
            output_param_count=get_target_params_count(target_dim, hyper_arch, output_dim),
            hop_id=i
        ) for i in range(num_hops)]
        self.hop_target_nets = [DynamicTargetNetwork(
            arch_list=hyper_arch,
            output_dim=output_dim,
            hop_id=i
        ) for i in range(num_hops)]
        
    def call(self, inputs, return_intermediate=False):
        # === STEP 1: Base Encoding ===
        z_base = self.enc(inputs)
        current_q = z_base
        intermediate_queries = [z_base]
        hop_data = []  # Store per-hop data for EDA
        
        # === STEP 2: Multi-Hop with 1:1 CNN + Hypernetwork Per Hop ===
        for i in range(self.num_hops):
            # --- 2a. CNN Residual (QUERY ENCODER mechanism) ---
            cnn_delta = self.hop_cnns[i](inputs)
            q_after_cnn = current_q + cnn_delta
            q_after_cnn = tf.linalg.l2_normalize(q_after_cnn, axis=1)
            
            # --- 2b. Centroid Context Lookup (HYPER mechanism) ---
            z_norm = tf.linalg.l2_normalize(q_after_cnn, axis=1)
            c_norm = tf.linalg.l2_normalize(CENTROID_VECS, axis=1)
            sims = tf.matmul(z_norm, c_norm, transpose_b=True)
            best_idx = tf.argmax(sims, axis=-1)  # Shape: [batch_size]
            ctx_vec = tf.gather(CENTROID_VECS, best_idx)  # Centroid context vector
            
            # --- 2c. Hypernetwork Weight Generation (HYPER mechanism) ---
            gen_params = self.hop_hypernets[i](ctx_vec)
            
            # --- 2d. Apply Generated Weights via Dynamic Target Net ---
            refined_delta = self.hop_target_nets[i](q_after_cnn, gen_params)
            current_q = q_after_cnn + refined_delta
            current_q = tf.linalg.l2_normalize(current_q, axis=1)
            
            if return_intermediate:
                intermediate_queries.append(current_q)
                hop_data.append({
                    'hop_id': i,
                    'centroid_indices': best_idx.numpy(),
                    'hyper_params_mean': np.mean(gen_params.numpy()),
                    'hyper_params_std': np.std(gen_params.numpy())
                })
        
        final_q = current_q
        
        # === STEP 3: Joystick Trick Prototype Attention (QUERY ENCODER mechanism) ===
        # Find nearest centroid for joystick position
        c_norm_final = tf.linalg.l2_normalize(CENTROID_VECS, axis=1)
        sims_final = tf.matmul(final_q, c_norm_final, transpose_b=True)
        best_idx_final = tf.argmax(sims_final, axis=-1)
        nearest_centroid = tf.gather(CENTROID_VECS, best_idx_final)
        
        # Compute query direction (the "swipe" from centroid)
        query_direction = final_q - nearest_centroid
        query_direction = tf.linalg.l2_normalize(query_direction, axis=1)
        
        # Compare against GLOBAL KNN directions (action offsets)
        directional_by_centroid = tf.reshape(MEM_BANK_VECS, [NUM_VISUAL_CENTROIDS, NUM_ACTIONS, EMBEDDING_DIM])
        prototype_directions = tf.gather(directional_by_centroid, best_idx_final)  # [batch, 10, 128]
        
        # Cosine similarity for direction matching
        sim_matrix = tf.einsum('bd,bkd->bk', query_direction, prototype_directions)  # [batch, 10]
        
        values, local_indices = tf.math.top_k(sim_matrix, k=min(NUM_NEIGHBORS, NUM_ACTIONS))
        attn_weights = tf.nn.softmax(values, axis=1)
        
        # Map local indices back to global prototype indices
        global_indices = tf.cast(best_idx_final[:, tf.newaxis], tf.int64) * NUM_ACTIONS + tf.cast(local_indices, tf.int64)
        
        # Gather labels using global indices
        neighbor_labels = tf.gather(MEM_BANK_LABELS, global_indices, axis=0)  # [batch, 10, 10]
        pred_retrieval = tf.reduce_sum(
            tf.expand_dims(attn_weights, -1) * neighbor_labels, 
            axis=1
        )
        
        if return_intermediate:
            return pred_retrieval, intermediate_queries, final_q, hop_data
        else:
            return pred_retrieval


class HyperQueryEncoderSystem(Model):
    """
    Full system orchestrating:
    1. Multi-Hop Hyper Retriever Branch (1:1 CNN + Hypernetwork per hop)
    2. Value Encoder Branch (Frozen Classifier)
    3. Direct Fusion
    """
    def __init__(self, retriever, value_encoder_path):
        super().__init__()
        self.retriever = retriever
        
        print(f"Loading Value Encoder from {value_encoder_path}...")
        self.value_encoder = models.load_model(value_encoder_path)
        self.value_encoder.trainable = False 
        print("Value Encoder Loaded & Frozen.")
        
    def call(self, inputs, training=False):
        pred_ret = self.retriever(inputs)
        pred_ve = tf.stop_gradient(self.value_encoder(inputs))
        
        # Direct fusion (can adjust weighting)
        final_pred = pred_ret  # * 0.7 + pred_ve * 0.3
        
        return final_pred


# ---------------------------------------------------------
# 4. INSTANTIATION & SETUP
# ---------------------------------------------------------

# Wrap the loaded encoder in the FrozenEncoderLayer
frozen_enc_layer = FrozenEncoderLayer(loaded_encoder)

# Create Multi-Hop Hyper Retriever System (1:1 CNN + Hypernetwork per hop)
retriever_branch = MultiHopHyperRetriever(
    enc=frozen_enc_layer, 
    num_hops=NUM_HOPS, 
    target_dim=EMBEDDING_DIM, 
    hyper_arch=TARGET_NET_ARCH,
    output_dim=EMBEDDING_DIM
)

print(f"\nInitialized Multi-Hop Hyper Retriever with 1:1 Ratio:")
print(f"  - {NUM_HOPS} Hop CNNs (Residual blocks)")
print(f"  - {NUM_HOPS} Hypernetworks (one per hop)")
print(f"  - {NUM_HOPS} Dynamic Target Networks (one per hop)")
print(f"  - Each Hypernetwork generates {TOTAL_PARAMS_PER_HOP} parameters")
print(f"  - Total generated parameters per forward pass: {TOTAL_PARAMS_PER_HOP * NUM_HOPS}")
print(f"  - Memory Bank: {NUM_PROTOTYPES} directional prototype vectors")
print(f"  - {NUM_VISUAL_CENTROIDS} visual centroids × {NUM_ACTIONS} action directions")
print(f"=== Combined: Multi-Hop (1:1) + Joystick Trick + Hypernetwork Per Hop ===")

# Create Full System
system_model = HyperQueryEncoderSystem(retriever_branch, VALUE_ENC_PATH)

optimizer = Adam(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)


# ---------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------

print("\n_______________________________________________________________________")
print(f"Starting Training: {NUM_HOPS} Hops with 1:1 CNN + Hypernetwork Per Hop")
print("_______________________________________________________________________")

dataset = tf.data.Dataset.from_tensor_slices((X_tr, y_tr_hot)).shuffle(10000).batch(BATCH_SIZE)

history_loss = []
history_acc = []

# EDA: Store query distributions for manifold visualization
eda_queries = []

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    
    for step, (x_batch, y_true) in enumerate(dataset):
        with tf.GradientTape() as tape:
            y_pred_prob = system_model(x_batch, training=True)
            loss_val = loss_fn(y_true, y_pred_prob)
            
        grads = tape.gradient(loss_val, system_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, system_model.trainable_weights))
        
        epoch_loss_avg.update_state(loss_val)
        epoch_acc_metric.update_state(y_true, y_pred_prob)
        
        if step % 50 == 0:
            print(f"Step {step}: Loss = {loss_val.numpy():.4f}, Acc = {epoch_acc_metric.result().numpy():.4f}")
    
    history_loss.append(epoch_loss_avg.result().numpy())
    history_acc.append(epoch_acc_metric.result().numpy())
    
    # EDA: Save query snapshots for manifold visualization
    if ENABLE_CONSOLIDATION_EDA and (epoch + 1) % 3 == 0:
        _, intermediate_qs, final_qs, hop_data = retriever_branch(X_te[:1000], return_intermediate=True)
        eda_queries.append({
            'epoch': epoch + 1,
            'queries': intermediate_qs[-1].numpy(),
            'final_queries': final_qs.numpy(),
            'hop_data': hop_data  # Per-hop centroid assignments and hypernetwork stats
        })
        print(f">>> EDA: Saved query snapshot for epoch {epoch+1}")
        for hd in hop_data:
            print(f"    Hop {hd['hop_id']}: Centroid mean={hd['centroid_indices'].mean():.2f}, Hyper params μ={hd['hyper_params_mean']:.4f} σ={hd['hyper_params_std']:.4f}")
    
    print(f">>> End Epoch: Avg Loss = {history_loss[-1]:.4f}, Avg Acc = {history_acc[-1]:.4f}")


# ---------------------------------------------------------
# 6. EVALUATION & EXPORT
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("Evaluation Results")
print("_______________________________________________________________________")

print("Calculating Test Set Metrics...")

y_ve_logits = system_model.value_encoder.predict(X_te)
y_ve_cls = np.argmax(y_ve_logits, axis=1)

y_final_probs = system_model.predict(X_te)
y_final_cls = np.argmax(y_final_probs, axis=1)

acc_ve = accuracy_score(y_te_int, y_ve_cls)
acc_final = accuracy_score(y_te_int, y_final_cls)

print(f"Value Encoder (Baseline) Accuracy : {acc_ve:.4f}")
print(f"Multi-Hop Hyper Joystick Accuracy: {acc_final:.4f}")

if acc_final > acc_ve:
    improvement = (acc_final - acc_ve) * 100
    print(f"(✓ Multi-Hop Hyper Joystick improved results by {improvement:.2f}%)")
else:
    print(f"(Note: May require more epochs or hyperparameter tuning.)")

print("\nClassification Report:")
print(classification_report(y_te_int, y_final_cls))

# Save the trained component
print("\nSaving Trained Multi-Hop Hyper Retriever...")
tf.saved_model.save(retriever_branch, SAVE_PATH_HQE_SYSTEM)
print(f"Saved to: {SAVE_PATH_HQE_SYSTEM}")

# Save EDA data for manifold visualization
if ENABLE_CONSOLIDATION_EDA and eda_queries:
    print("\nSaving EDA Manifold Snapshots...")
    os.makedirs(EDA_SAVE_PATH, exist_ok=True)
    import pickle
    with open(os.path.join(EDA_SAVE_PATH, "query_evolution.pkl"), "wb") as f:
        pickle.dump(eda_queries, f)
    print(f"Saved EDA data to: {EDA_SAVE_PATH}")
    print("(Use this data to create t-SNE plots showing manifold evolution per hop)")

print("\n_______________________________________________________________________")
print("Training Complete!")
print("_______________________________________________________________________")