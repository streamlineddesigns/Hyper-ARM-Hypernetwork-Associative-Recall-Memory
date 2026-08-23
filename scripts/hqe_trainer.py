# ---------------------------------------------------------
# HYPER QUERY ENCODER SCRIPT (UPDATED VERSION)
# Multi-Hop + Hypernetwork Per Hop + Learnable Temperature + STM
# LTM Seeding Added (Identical to STM Strategy 1)
# ---------------------------------------------------------

# ---------------------------------------------------------
# SQLITE FIX: Must be at the TOP of the script
# ---------------------------------------------------------
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    print("ERROR: pysqlite3-binary not installed! Run 'pip install pysqlite3-binary'")
    exit()

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
import os
import sys
import ast 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import chromadb
import uuid
import time
import pickle

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# DB Paths (From Script A - Original Names)
CHROMA_DB_PATH = "./chroma_db_mnist"
COLLECTION_NAME = "mnist_sparse_collection"

# Short Term Memory Configuration (From Script A)
LOGGING_STM = True
USING_STM = True                  
STM_DB_PATH = "./chroma_db_stm"   
STM_COLLECTION_NAME = "stm_collection"
SIMILARITY_THRESHOLD = 0.65

# Model Paths (From Script A)
ENCODER_PATH = "./saved_cnne_model_dir"
VALUE_ENC_PATH = "./saved_mnist_classifier_dir" 
SAVE_PATH_HQE_SYSTEM = "./saved_hqe_hyper_multi_hop_system"

# Embedding & Architecture (Script A dims + Script B Hypernetwork)
EMBEDDING_DIM = 128 
NUM_NEIGHBORS = 5       
BATCH_SIZE = 128
EPOCHS = 10             
LEARNING_RATE = 0.003

# Multi-Hop Configuration (From Script A)
NUM_HOPS = 2            

# Temperature Config (From Script A)
MIN_TEMP = 0.5
MAX_TEMP = 2.0
INIT_TEMP = 1.0 

# STM Optimization Config (From Script A)
STM_INSERT_BATCH_SIZE = 32
STM_OPTIMIZATION_SUBSET_RATIO = 0.2
STM_PATIENCE = 5000

# STM Candidate Retrieval Config (From Script A)
STM_LTM_MIN_SIM = 0.75
STM_DEDUP_CANDIDATES = True
STM_SORT_MODE = "similarity"

# Hybrid Strategy Flags (From Script A)
HYBRID_USE_LOW_SIM = True
HYBRID_USE_LTM_PROTO = True

# Hypernetwork Config (From Script B)
NUM_VISUAL_CENTROIDS = 64
NUM_ACTIONS = 10
TARGET_NET_ARCH = [64, 32]
HYPER_INTERMEDIATE_DIM = 98

# EDA Config (From Script B)
ENABLE_CONSOLIDATION_EDA = False
EDA_SAVE_PATH = "./eda_manifold_snapshots"

# ---------------------------------------------------------
# HELPER: Robust SavedModel Caller (From Script B)
# ---------------------------------------------------------
def _extract_tensor(out):
    if isinstance(out, dict):
        return list(out.values())[0]
    if isinstance(out, (list, tuple)):
        return out[0]
    return out

def call_frozen_encoder(module, x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    for kwargs in ({}, {"training": False}):
        try:
            return _extract_tensor(module(x, **kwargs) if kwargs else module(x))
        except Exception:
            pass
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
# 1. DATA PREPARATION (From Script A - 50/50 Split)
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

# SINGLE SPLIT: 50% Train/Val, 50% Test (From Script A)
idx_train_val, idx_test, _, _ = train_test_split(indices, Y_full, test_size=0.5, stratify=Y_full)

X_train_val = X_processed[idx_train_val]
y_train_val_hot = Y_onehot[idx_train_val]
y_train_val_int = Y_full[idx_train_val]

X_te = X_processed[idx_test]
y_te_int = Y_full[idx_test]
y_te_hot = Y_onehot[idx_test]

# Shuffle training data
shuffle_idx = np.random.permutation(len(X_train_val))
X_train_val = X_train_val[shuffle_idx]
y_train_val_hot = y_train_val_hot[shuffle_idx]
y_train_val_int = y_train_val_int[shuffle_idx]

# ---------------------------------------------------------
# 2. Load Frozen Encoder (Needed for LTM Seeding)
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("Loading Frozen Encoder for LTM Seeding")
print("_______________________________________________________________________")
print(f"Loading Frozen Encoder from {ENCODER_PATH}...")
loaded_encoder = tf.saved_model.load(ENCODER_PATH)
print("Frozen Encoder loaded successfully.")

# ---------------------------------------------------------
# 3. LTM SEEDING (Identical to STM Strategy 1 - Z's Only)
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("LTM Seeding: Identical to STM Strategy 1 (Pre-Training)")
print("_______________________________________________________________________")

# Step 1: Encode All Training Data
print("Encoding Training Data for LTM Seed...")
Z_pool = encode_images(loaded_encoder, X_train_val, batch_size=256)
Z_pool_norm = tf.nn.l2_normalize(Z_pool, axis=1).numpy()

# Step 2: Split (Identical Ratio to STM - 20% Val, 80% Candidates)
n_total = len(Z_pool_norm)
n_val = int(n_total * STM_OPTIMIZATION_SUBSET_RATIO)  # 20%
shuffle_indices = np.random.permutation(n_total)
val_indices = shuffle_indices[:n_val]
candidate_indices = shuffle_indices[n_val:]

Z_val = Z_pool_norm[val_indices]
Y_val_int = y_train_val_int[val_indices]

Z_candidates = Z_pool_norm[candidate_indices]
Y_candidates_int = y_train_val_int[candidate_indices]
Y_candidates_hot = Y_onehot[idx_train_val][candidate_indices]

print(f"Z_pool: {n_total} | Z_val (20%): {n_val} | Z_candidates (80%): {len(Z_candidates)}")

# Step 3: Group by Label
print("Grouping candidates by label...")
label_groups = {}
for i in range(NUM_ACTIONS):
    mask = (Y_candidates_int == i)
    label_groups[i] = {
        'vecs': Z_candidates[mask],
        'labels_int': Y_candidates_int[mask],
        'labels_hot': Y_candidates_hot[mask]
    }
    print(f"  Label {i}: {len(Z_candidates[mask])} candidates")

# Step 4: Initialize LTM Storage
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass
collection = client.get_or_create_collection(COLLECTION_NAME)

current_ltm_vecs = []  # List of arrays
current_ltm_labels = []  # List of arrays
best_acc = 0.0
global_insert_count = 0

# Step 5: k-NN Accuracy Function (Frozen Encoder Retrieval)
def knn_accuracy(query_zs, query_labels, memory_zs, memory_labels, k=5):
    if len(memory_zs) == 0:
        return 0.0
    
    # Ensure proper 2D shape
    if isinstance(memory_zs, list):
        memory_zs_arr = np.vstack(memory_zs)
    else:
        memory_zs_arr = memory_zs
    
    if isinstance(memory_labels, list):
        memory_labels_arr = np.concatenate(memory_labels)
    else:
        memory_labels_arr = memory_labels
    
    sims = np.dot(query_zs, memory_zs_arr.T)
    top_k_idx = np.argsort(sims, axis=1)[:, -k:]
    
    preds = []
    for i in range(len(query_zs)):
        neighbor_labels = memory_labels_arr[top_k_idx[i]]
        label_counts = np.bincount(neighbor_labels.astype(int), minlength=NUM_ACTIONS)
        pred = np.argmax(label_counts)
        preds.append(pred)
    
    return accuracy_score(query_labels, preds)

# Step 6: Collect ALL Batches First (Grouped by Label, Then Shuffled)
print("\nCollecting and Shuffling Batches Across All Labels...")
all_batches = []

for label in range(NUM_ACTIONS):
    group = label_groups[label]
    group_vecs = group['vecs']
    group_labels_int = group['labels_int']
    group_labels_hot = group['labels_hot']
    
    # Process each label to create batches (but don't accept/reject yet)
    while len(group_vecs) > 0:
        
        # === Surprise Gate (Hard Filter) ===
        if len(current_ltm_vecs) == 0:
            # First pass: all candidates pass (will re-filter during actual processing)
            eligible_mask = np.ones(len(group_vecs), dtype=bool)
            eligible_sims = np.zeros(len(group_vecs))
        else:
            # Calculate max similarity against current LTM
            ltm_arr = np.vstack(current_ltm_vecs)
            sims = np.dot(group_vecs, ltm_arr.T)
            max_sims = np.max(sims, axis=1)
            
            # Hard Filter: sim < 0.65
            eligible_mask = (max_sims < SIMILARITY_THRESHOLD)
            eligible_sims = max_sims
        
        eligible_vecs = group_vecs[eligible_mask]
        eligible_sims = eligible_sims[eligible_mask]
        eligible_labels_int = group_labels_int[eligible_mask]
        eligible_labels_hot = group_labels_hot[eligible_mask]
        
        if len(eligible_vecs) == 0:
            break
        
        # === Sort Ascending by Similarity (Lowest First) ===
        sort_idx = np.argsort(eligible_sims)
        eligible_vecs = eligible_vecs[sort_idx]
        eligible_sims = eligible_sims[sort_idx]
        eligible_labels_int = eligible_labels_int[sort_idx]
        eligible_labels_hot = eligible_labels_hot[sort_idx]
        
        # === Batch (Size 32) ===
        batch_size = min(STM_INSERT_BATCH_SIZE, len(eligible_vecs))
        batch_vecs = eligible_vecs[:batch_size]
        batch_labels_int = eligible_labels_int[:batch_size]
        batch_labels_hot = eligible_labels_hot[:batch_size]
        batch_sims = eligible_sims[:batch_size]
        
        # Store batch for later shuffled processing
        all_batches.append({
            'vecs': batch_vecs,
            'labels_int': batch_labels_int,
            'labels_hot': batch_labels_hot,
            'sims': batch_sims,
            'label': label
        })
        
        # Remove processed from pool
        group_vecs = eligible_vecs[batch_size:]
        group_labels_int = eligible_labels_int[batch_size:]
        group_labels_hot = eligible_labels_hot[batch_size:]

print(f">>> Total Batches Collected: {len(all_batches)}")

# === SHUFFLE ALL BATCHES ===
np.random.shuffle(all_batches)
print(">>> Batches Shuffled (Cross-Label)")

# Step 7: Process Shuffled Batches (Accept/Reject)
print("\nStarting LTM Seeding Loop (Shuffled Batch Order)...")
start_time = time.time()

patience_counter = 0
batch_idx = 0

for batch_data in all_batches:
    if patience_counter >= STM_PATIENCE:
        print(f"  >>> STM Patience Reached. Stopping Optimization.")
        break
    
    batch_vecs = batch_data['vecs']
    batch_labels_int = batch_data['labels_int']
    batch_labels_hot = batch_data['labels_hot']
    batch_label = batch_data['label']
    
    # === Re-apply Surprise Gate (LTM may have grown since collection) ===
    if len(current_ltm_vecs) > 0:
        ltm_arr = np.vstack(current_ltm_vecs)
        sims = np.dot(batch_vecs, ltm_arr.T)
        max_sims = np.max(sims, axis=1)
        
        # Filter out any vectors that are now redundant
        keep_mask = (max_sims < SIMILARITY_THRESHOLD)
        if np.sum(keep_mask) == 0:
            patience_counter += 1
            continue
        
        # Keep only eligible vectors
        batch_vecs = batch_vecs[keep_mask]
        batch_labels_int = batch_labels_int[keep_mask]
        batch_labels_hot = batch_labels_hot[keep_mask]
        
        if len(batch_vecs) == 0:
            continue
    
    # === Test & Accept ===
    # Temp Add
    temp_ltm_vecs = current_ltm_vecs + [batch_vecs]
    temp_ltm_vecs_arr = np.vstack(temp_ltm_vecs)
    temp_ltm_labels = current_ltm_labels + [batch_labels_int]
    temp_ltm_labels_arr = np.concatenate(temp_ltm_labels)
    
    # Validate on ALL Z_val (mixed classes)
    acc = knn_accuracy(
        Z_val, 
        Y_val_int, 
        temp_ltm_vecs_arr, 
        temp_ltm_labels_arr, 
        k=NUM_NEIGHBORS
    )
    
    if acc > best_acc:
        # ACCEPT
        best_acc = acc
        current_ltm_vecs.append(batch_vecs)
        current_ltm_labels.append(batch_labels_int)
        
        # Insert into Chroma
        ids_to_insert = [f"ltm_seed_{global_insert_count + j}" for j in range(len(batch_vecs))]
        metadatas_to_insert = []
        for idx in range(len(batch_vecs)):
            gt_vec = [0]*NUM_ACTIONS
            gt_vec[int(batch_labels_int[idx])] = 1
            metadatas_to_insert.append({
                "true_label": int(batch_labels_int[idx]), 
                "one_hot_vector": str(gt_vec)
            })
        
        collection.add(
            embeddings=batch_vecs.tolist(),
            ids=ids_to_insert,
            metadatas=metadatas_to_insert
        )
        
        global_insert_count += len(batch_vecs)
        patience_counter = 0
        print(f"  Batch {batch_idx+1} ACCEPTED (Label {batch_label}). New Acc: {acc:.4f} | Total LTM: {global_insert_count}")
    else:
        # REJECT
        patience_counter += 1
        print(f"  Batch {batch_idx+1} REJECTED (Label {batch_label}). Acc: {acc:.4f} | Patience: {patience_counter}/{STM_PATIENCE}")
    
    batch_idx += 1

end_time = time.time()
print(f"\n>>> LTM Seeding Finished in {end_time - start_time:.2f} seconds.")
print(f">>> Total Vectors Inserted into LTM: {global_insert_count}")
print(f">>> Final LTM Accuracy (on Val Subset): {best_acc:.4f}")

# ---------------------------------------------------------
# 4. Load LTM for Training (From Script A)
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("Loading Seeded LTM for Training")
print("_______________________________________________________________________")

results = collection.get(include=['embeddings', 'metadatas'])
db_vecs_raw = np.array(results['embeddings']).astype('float32')

db_labels_raw = []
for m in results['metadatas']:
    try: 
        db_labels_raw.append(ast.literal_eval(m['one_hot_vector']))
    except: 
        db_labels_raw.append([0]*NUM_ACTIONS) 
        
db_labels_raw = np.array(db_labels_raw).astype('float32')
MEM_BANK_VECS = tf.constant(db_vecs_raw)
MEM_BANK_LABELS = tf.constant(db_labels_raw)

print(f"LTM Loaded: {len(db_vecs_raw)} vectors")

# ---------------------------------------------------------
# 5. Generate Visual Centroids for Hypernetwork (From Script B)
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("Generating Visual Centroids for Hypernetwork Context")
print("_______________________________________________________________________")

# Generate latent vectors for training data
print("Encoding training data for K-Means...")
Z_train_val = encode_images(loaded_encoder, X_train_val, batch_size=256)
print(f"Z_train shape: {Z_train_val.shape}")

# K-Means to find Visual Centroids
print(f"Running K-Means with K={NUM_VISUAL_CENTROIDS}...")
kmeans = KMeans(n_clusters=NUM_VISUAL_CENTROIDS, random_state=42, n_init=10)
kmeans.fit(Z_train_val)
VISUAL_CENTROIDS = kmeans.cluster_centers_.astype('float32')
CENTROID_VECS = tf.constant(VISUAL_CENTROIDS)
print(f"Generated {NUM_VISUAL_CENTROIDS} visual centroids.")

# ---------------------------------------------------------
# 6. Short Term Memory Bank (STM) - Initialization (From Script A)
# ---------------------------------------------------------
if USING_STM:
    print(f"Initializing Short Term Memory DB at {STM_DB_PATH}...")
    stm_client = chromadb.PersistentClient(path=STM_DB_PATH)
    try: stm_client.delete_collection(STM_COLLECTION_NAME)
    except: pass
    stm_collection = stm_client.get_or_create_collection(STM_COLLECTION_NAME)
    stm_vecs_list = []   
    stm_labels_list = [] 
else:
    stm_collection = None

# ---------------------------------------------------------
# 7. ARCHITECTURE DEFINITIONS
# ---------------------------------------------------------

class FrozenEncoderLayer(layers.Layer):
    """Robust Frozen Encoder Layer (From Script B)"""
    def __init__(self, module, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self.trainable = False
        
    def call(self, inputs, training=False):
        res = call_frozen_encoder(self.module, inputs)
        return tf.reshape(res, [tf.shape(res)[0], -1]) 


class ResidualCNN(keras.Model):
    """Residual CNN per hop (From Script B)"""
    def __init__(self, target_dim, hop_id=0):
        super().__init__()
        self.hop_id = hop_id
        name_prefix = f"hop{hop_id}"
        
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                                   name=f"{name_prefix}_conv1", kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn1 = layers.BatchNormalization(name=f"{name_prefix}_bn1")
        self.drop1 = layers.Dropout(0.3, name=f"{name_prefix}_drop1")
        self.pool1 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool1")
        
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                                   name=f"{name_prefix}_conv2", kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn2 = layers.BatchNormalization(name=f"{name_prefix}_bn2")
        self.drop2 = layers.Dropout(0.3, name=f"{name_prefix}_drop2")
        
        self.flatten = layers.Flatten(name=f"{name_prefix}_flatten")
        self.dense_proj = layers.Dense(target_dim, activation='relu', name=f"{name_prefix}_dense", kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.out_layer = layers.Dense(target_dim, activation='linear', name=f"{name_prefix}_out", kernel_regularizer=tf.keras.regularizers.l2(1e-4)) 

    def call(self, raw_image_inputs, training=None):
        x = self.conv1(raw_image_inputs)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)
        x = self.flatten(x)
        x = self.dense_proj(x)
        delta_z = self.out_layer(x)
        return delta_z


def get_target_params_count(input_dim, arch_list, output_dim):
    """Calculate total parameters needed for the generated target network."""
    count = 0
    prev = input_dim
    for size in arch_list:
        count += prev * size + size
        prev = size
    count += prev * output_dim + output_dim
    return count

TOTAL_PARAMS_PER_HOP = get_target_params_count(EMBEDDING_DIM, TARGET_NET_ARCH, EMBEDDING_DIM)


class CentroidHypernetwork(keras.Model):
    """Hypernetwork that generates weights based on centroid context (From Script B)"""
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
    """Executes MLP defined by generated params (From Script B)"""
    def __init__(self, arch_list, output_dim, hop_id=0):
        super().__init__()
        self.hop_id = hop_id
        self.arch_list = arch_list
        self.output_dim = output_dim
        
    def call(self, hop_query, generated_params):
        current_x = hop_query
        current_dim = tf.shape(current_x)[-1] 
        offset = 0
        
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
            
        out_dim = self.output_dim
        w_size_out = current_dim * out_dim
        b_size_out = out_dim
        
        w_flat_out = generated_params[:, offset : offset + w_size_out]
        b_val_out  = generated_params[:, offset + w_size_out : offset + w_size_out + b_size_out]
        
        w_matrix_out = tf.reshape(w_flat_out, [tf.shape(current_x)[0], current_dim, out_dim])
        refined_delta = tf.einsum('bi,bij->bj', current_x, w_matrix_out) + b_val_out
        
        return refined_delta


class MultiHopHyperRetriever(Model):
    """
    Multi-Hop with 1:1 CNN + Hypernetwork Per Hop
    Retrieval uses Direct Cosine Similarity (From Script A)
    Learnable Temperature (From Script A)
    """
    def __init__(self, enc, num_hops, target_dim, hyper_arch, output_dim, initial_temperature=1.0):
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
        
        # Learnable Temperature (From Script A)
        self.log_temp = tf.Variable(
            np.log(initial_temperature), 
            trainable=True, 
            dtype=tf.float32, 
            name="learnable_log_temperature"
        )
        
    def get_temperature(self):
        temp = tf.exp(self.log_temp)
        return tf.clip_by_value(temp, MIN_TEMP, MAX_TEMP)
        
    def call(self, inputs, training=None, stm_vecs=None, stm_labels=None, return_sim=False, return_intermediate=False):
        # === STEP 1: Base Encoding ===
        z_base = self.enc(inputs, training=training)
        current_q = z_base
        intermediate_queries = [z_base]
        hop_data = []
        
        # === STEP 2: Multi-Hop with 1:1 CNN + Hypernetwork Per Hop ===
        for i in range(self.num_hops):
            # CNN Residual
            cnn_delta = self.hop_cnns[i](inputs, training=training)
            q_after_cnn = current_q + cnn_delta
            q_after_cnn = tf.linalg.l2_normalize(q_after_cnn, axis=1)
            
            # Centroid Context Lookup
            z_norm = tf.linalg.l2_normalize(q_after_cnn, axis=1)
            c_norm = tf.linalg.l2_normalize(CENTROID_VECS, axis=1)
            sims = tf.matmul(z_norm, c_norm, transpose_b=True)
            best_idx = tf.argmax(sims, axis=-1)
            ctx_vec = tf.gather(CENTROID_VECS, best_idx)
            
            # Hypernetwork Weight Generation
            gen_params = self.hop_hypernets[i](ctx_vec)
            
            # Apply Generated Weights
            refined_delta = self.hop_target_nets[i](q_after_cnn, gen_params)
            current_q = q_after_cnn + refined_delta
            current_q = tf.linalg.l2_normalize(current_q, axis=1)
            
            if return_intermediate:
                intermediate_queries.append(current_q)
                hop_data.append({
                    'hop_id': i,
                    'centroid_indices': best_idx.numpy() if hasattr(best_idx, 'numpy') else best_idx,
                    'hyper_params_mean': float(np.mean(gen_params.numpy() if hasattr(gen_params, 'numpy') else gen_params)),
                    'hyper_params_std': float(np.std(gen_params.numpy() if hasattr(gen_params, 'numpy') else gen_params))
                })
        
        final_q = current_q
        final_q = tf.nn.l2_normalize(final_q, axis=1)
        
        # Add noise during training (From Script A)
        if training:
            noise = tf.random.normal(shape=tf.shape(final_q), mean=0.0, stddev=0.01)
            final_q = final_q + noise
        
        # === STEP 3: Direct Cosine Similarity Retrieval (From Script A) ===
        main_vecs_norm = tf.nn.l2_normalize(MEM_BANK_VECS, axis=1)
        sim_matrix_main = tf.matmul(final_q, main_vecs_norm, transpose_b=True)
        values_main, indices_main = tf.math.top_k(sim_matrix_main, k=NUM_NEIGHBORS)
        max_sim_main = tf.reduce_max(values_main, axis=1)
        
        current_temp = self.get_temperature()
        scaled_values_main = values_main / current_temp 
        attn_weights_main = tf.nn.softmax(scaled_values_main, axis=1)
        neighbor_labels_main = tf.gather(MEM_BANK_LABELS, indices_main) 
        pred_main = tf.reduce_sum(tf.expand_dims(attn_weights_main, -1) * neighbor_labels_main, axis=1)
        
        pred_final = pred_main
        
        # === STEP 4: STM Retrieval (From Script A) ===
        if stm_vecs is not None and tf.shape(stm_vecs)[0] > 0:
            stm_vecs_norm = tf.nn.l2_normalize(stm_vecs, axis=1)
            sim_matrix_stm = tf.matmul(final_q, stm_vecs_norm, transpose_b=True)
            k_stm = tf.minimum(NUM_NEIGHBORS, tf.shape(stm_vecs_norm)[0])
            values_stm, indices_stm = tf.math.top_k(sim_matrix_stm, k=k_stm)
            scaled_values_stm = values_stm / current_temp 
            attn_weights_stm = tf.nn.softmax(scaled_values_stm, axis=1)
            neighbor_labels_stm = tf.gather(stm_labels, indices_stm) 
            pred_stm = tf.reduce_sum(tf.expand_dims(attn_weights_stm, -1) * neighbor_labels_stm, axis=1)
            # Weighted Average (From Script A)
            pred_final = (pred_main * 0.7 + pred_stm * 0.3)
        
        if return_intermediate:
            if return_sim:
                return {'predictions': pred_final, 'max_similarity': max_sim_main}, intermediate_queries, final_q, hop_data
            return pred_final, intermediate_queries, final_q, hop_data
        else:
            if return_sim:
                return {'predictions': pred_final, 'max_similarity': max_sim_main}
            return pred_final


class GuidedSystem(Model):
    """Full system with Data Augmentation (From Script A)"""
    def __init__(self, retriever, value_encoder_path):
        super().__init__()
        self.retriever = retriever
        
        self.data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomTranslation(0.01, 0.01),
                layers.RandomZoom(0.01),
                layers.RandomContrast(0.01),
            ],
            name="data_augmentation"
        )
        
        print(f"Loading Value Encoder from {value_encoder_path}...")
        self.value_encoder = models.load_model(value_encoder_path)
        self.value_encoder.trainable = False 
        print("Value Encoder Loaded & Frozen.")
        
    def call(self, inputs, training=None, **kwargs):
        if training:
            inputs = self.data_augmentation(inputs, training=True)
        pred_ret = self.retriever(inputs, training=training, **kwargs)
        return pred_ret

# ---------------------------------------------------------
# 8. INSTANTIATION & SETUP
# ---------------------------------------------------------
frozen_enc_layer = FrozenEncoderLayer(loaded_encoder)

retriever_branch = MultiHopHyperRetriever(
    enc=frozen_enc_layer, 
    num_hops=NUM_HOPS, 
    target_dim=EMBEDDING_DIM, 
    hyper_arch=TARGET_NET_ARCH,
    output_dim=EMBEDDING_DIM,
    initial_temperature=INIT_TEMP
)

print(f"\nInitialized Multi-Hop Hyper Retriever:")
print(f"  - {NUM_HOPS} Hop CNNs (Residual blocks)")
print(f"  - {NUM_HOPS} Hypernetworks (one per hop)")
print(f"  - {NUM_HOPS} Dynamic Target Networks (one per hop)")
print(f"  - Each Hypernetwork generates {TOTAL_PARAMS_PER_HOP} parameters")
print(f"  - Learnable Temperature: {INIT_TEMP}")
print(f"  - LTM Vectors: {len(db_vecs_raw)}")
print(f"  - Visual Centroids: {NUM_VISUAL_CENTROIDS}")

system_model = GuidedSystem(retriever_branch, VALUE_ENC_PATH)
system_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE), 
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
    metrics=['accuracy']
)

# ---------------------------------------------------------
# 9. TRAINING LOOP (From Script A - Keras fit + Callbacks)
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print(f"Starting Training: Multi-Hop ({NUM_HOPS}) + Hypernetwork + Learnable Temperature")
print("_______________________________________________________________________")

class TemperatureLogger(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        temp = self.model.retriever.get_temperature().numpy()
        print(f" >>> Epoch {epoch+1}: Learned Temp = {temp:.3f}")

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=1, min_lr=1e-6, verbose=1)

# EDA Storage
eda_queries = []

class EDACallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if ENABLE_CONSOLIDATION_EDA and (epoch + 1) % 3 == 0:
            try:
                output, intermediate_qs, final_qs, hop_data = self.model.retriever(
                    X_te[:1000], training=False, return_intermediate=True
                )
                eda_queries.append({
                    'epoch': epoch + 1,
                    'queries': intermediate_qs[-1].numpy(),
                    'final_queries': final_qs.numpy(),
                    'hop_data': hop_data
                })
                print(f">>> EDA: Saved query snapshot for epoch {epoch+1}")
            except Exception as e:
                print(f">>> EDA Warning: Could not save snapshot: {e}")

history = system_model.fit(
    X_train_val, y_train_val_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_te, y_te_hot), 
    callbacks=[early_stop, reduce_lr, TemperatureLogger(), EDACallback()]
)

# ---------------------------------------------------------
# 10. EVALUATION & STM OPTIMIZATION (From Script A - 3 Pass System)
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("Evaluation Results (Three-Pass System)")
print("_______________________________________________________________________")

def calculate_accuracy_with_stm(model, X_subset, y_true_subset, stm_vecs_np, stm_labels_np):
    if len(stm_vecs_np) > 0:
        stm_v_tf = tf.constant(stm_vecs_np, dtype=tf.float32)
        stm_l_tf = tf.constant(stm_labels_np, dtype=tf.float32)
    else:
        stm_v_tf = None
        stm_l_tf = None
        
    preds = []
    batch_size = 256
    for i in range(0, len(X_subset), batch_size):
        x_b = X_subset[i:i+batch_size]
        out = model.retriever(x_b, training=False, stm_vecs=stm_v_tf, stm_labels=stm_l_tf)
        preds.extend(np.argmax(out.numpy(), axis=1))
    
    return accuracy_score(y_true_subset, preds)

# Prepare Optimization Subset
n_opt_samples = int(len(X_te) * STM_OPTIMIZATION_SUBSET_RATIO)
opt_indices = np.random.choice(len(X_te), n_opt_samples, replace=False)
X_opt = X_te[opt_indices]
y_opt_int = y_te_int[opt_indices]

# =========================================================
# PASS 1: HYBRID CANDIDATE IDENTIFICATION
# =========================================================
print("\n_______________________________________________________________________")
print("PASS 1: Hybrid Candidate Identification (Low-Sim + LTM Prototype)")
print("_______________________________________________________________________")

eval_dataset = tf.data.Dataset.from_tensor_slices((X_te, y_te_int, y_te_hot)).batch(BATCH_SIZE)

pass1_preds = []
pass1_trues = []
pass1_sims  = []

LTM_VECS_NORM = tf.nn.l2_normalize(MEM_BANK_VECS, axis=1)

ltm_hits = {}
strategy1_candidates = []
strategy2_candidates = []

n_errors_total      = 0
n_errors_low_sim    = 0
n_errors_ltm_proto  = 0
n_errors_no_match   = 0

for step, (x_batch, y_true_int, y_true_hot) in enumerate(eval_dataset):
    output = system_model(x_batch, training=False, stm_vecs=None, stm_labels=None, return_sim=True)
    pred_final = output['predictions']
    max_sim    = output['max_similarity'].numpy()

    y_pred_cls = np.argmax(pred_final.numpy(), axis=1)
    y_true_cls = y_true_int.numpy()

    pass1_preds.extend(y_pred_cls)
    pass1_trues.extend(y_true_cls)
    pass1_sims.extend(max_sim.tolist())

    if not USING_STM:
        continue

    is_wrong = (y_pred_cls != y_true_cls)
    wrong_idx = np.where(is_wrong)[0]
    n_errors_total += len(wrong_idx)
    if len(wrong_idx) == 0:
        continue

    x_wrong     = x_batch.numpy()[wrong_idx]
    y_hot_wrong = y_true_hot.numpy()[wrong_idx]
    y_int_wrong = y_true_int.numpy()[wrong_idx]
    sim_wrong   = max_sim[wrong_idx]

    # STRATEGY 1: Wrong + Low Similarity
    if HYBRID_USE_LOW_SIM:
        is_low_sim = (sim_wrong < SIMILARITY_THRESHOLD)
        low_sim_idx = np.where(is_low_sim)[0]
        
        if len(low_sim_idx) > 0:
            for r in low_sim_idx:
                strategy1_candidates.append({
                    'type': 'low_sim',
                    'image': x_wrong[r],
                    'label_hot': y_hot_wrong[r],
                    'label_int': int(y_int_wrong[r]),
                    'sim': float(sim_wrong[r])
                })
            n_errors_low_sim += len(low_sim_idx)

    # STRATEGY 2: Wrong + Found Correct LTM Prototype
    if HYBRID_USE_LTM_PROTO:
        z_query = frozen_enc_layer(x_wrong, training=False)
        z_query_norm = tf.nn.l2_normalize(z_query, axis=1)
        y_hot_wrong_tf = tf.constant(y_hot_wrong, dtype=tf.float32)

        sim_matrix = tf.matmul(z_query_norm, LTM_VECS_NORM, transpose_b=True)
        label_match = tf.matmul(y_hot_wrong_tf, MEM_BANK_LABELS, transpose_b=True)
        masked_sims = tf.where(label_match > 0.5, sim_matrix, tf.fill(tf.shape(sim_matrix), -1e9))

        best_idx = tf.argmax(masked_sims, axis=1)
        best_sim = tf.reduce_max(masked_sims, axis=1)
        
        best_idx_np = best_idx.numpy()
        best_sim_np = best_sim.numpy()

        for r in range(len(wrong_idx)):
            s = float(best_sim_np[r])
            
            if s < STM_LTM_MIN_SIM:
                n_errors_no_match += 1
                continue
            else:
                n_errors_ltm_proto += 1
            
            g_idx = int(best_idx_np[r])
            rec = ltm_hits.get(g_idx)
            if rec is None:
                ltm_hits[g_idx] = {'count': 1, 'best_sim': s, 'label_int': int(y_int_wrong[r])}
            else:
                rec['count'] += 1
                if s > rec['best_sim']:
                    rec['best_sim'] = s

acc_pass1 = accuracy_score(pass1_trues, pass1_preds)
print(f">>> PASS 1 Accuracy: {acc_pass1:.4f}")
print(f">>> Total Errors in Pass 1           : {n_errors_total}")
if HYBRID_USE_LOW_SIM:
    print(f">>> Strategy 1 (Low Sim) Candidates  : {len(strategy1_candidates)}")
if HYBRID_USE_LTM_PROTO:
    print(f">>> Strategy 2 (LTM Proto) Candidates: {len(ltm_hits)} unique prototypes")
    print(f">>> Errors covered by LTM Proto      : {n_errors_ltm_proto}")
    print(f">>> Errors with NO prototype > {STM_LTM_MIN_SIM:.2f}   : {n_errors_no_match}")

# =========================================================
# MERGE CANDIDATES FROM BOTH STRATEGIES
# =========================================================
print("\n_______________________________________________________________________")
print("Merging Hybrid Candidates")
print("_______________________________________________________________________")

candidate_vectors    = []
candidate_labels_hot = []
candidate_labels_int = []
candidate_sims       = []
candidate_counts     = []
candidate_sources    = []

if HYBRID_USE_LOW_SIM and len(strategy1_candidates) > 0:
    print(f"Encoding {len(strategy1_candidates)} Strategy 1 candidates...")
    strat1_imgs = np.array([c['image'] for c in strategy1_candidates])
    strat1_hots = np.array([c['label_hot'] for c in strategy1_candidates])
    strat1_ints = np.array([c['label_int'] for c in strategy1_candidates])
    strat1_sims = np.array([c['sim'] for c in strategy1_candidates])
    
    strat1_vecs = []
    encode_batch_size = 256
    for i in range(0, len(strat1_imgs), encode_batch_size):
        batch_imgs = strat1_imgs[i:i+encode_batch_size]
        batch_vecs = frozen_enc_layer(batch_imgs, training=False).numpy()
        strat1_vecs.append(batch_vecs)
    strat1_vecs = np.vstack(strat1_vecs)
    
    for j in range(len(strat1_vecs)):
        candidate_vectors.append(strat1_vecs[j])
        candidate_labels_hot.append(strat1_hots[j])
        candidate_labels_int.append(strat1_ints[j])
        candidate_sims.append(strat1_sims[j])
        candidate_counts.append(1)
        candidate_sources.append('low_sim')
    
    print(f">>> Strategy 1 encoded vectors: {len(strat1_vecs)}")

if HYBRID_USE_LTM_PROTO and len(ltm_hits) > 0:
    hit_indices = np.array(sorted(ltm_hits.keys()), dtype=np.int64)

    if not STM_DEDUP_CANDIDATES:
        expanded = []
        for g in hit_indices:
            expanded.extend([g] * ltm_hits[g]['count'])
        hit_indices = np.array(expanded, dtype=np.int64)

    gathered = tf.gather(MEM_BANK_VECS, hit_indices).numpy()
    g_labels = tf.gather(MEM_BANK_LABELS, hit_indices).numpy()

    for j, g in enumerate(hit_indices):
        rec = ltm_hits[int(g)]
        candidate_vectors.append(gathered[j])
        candidate_labels_hot.append(g_labels[j])
        candidate_labels_int.append(int(np.argmax(g_labels[j])))
        candidate_counts.append(rec['count'])
        candidate_sims.append(float(rec['best_sim']))
        candidate_sources.append('ltm_proto')
    
    print(f">>> Strategy 2 LTM prototypes: {len(hit_indices)}")

if STM_DEDUP_CANDIDATES and len(candidate_vectors) > 0:
    print("\nApplying cross-strategy deduplication...")
    cand_vecs_arr = np.array(candidate_vectors)
    
    unique_indices = []
    seen_hashes = set()
    
    for i in range(len(cand_vecs_arr)):
        vec_hash = tuple(np.round(cand_vecs_arr[i], decimals=4))
        if vec_hash not in seen_hashes:
            seen_hashes.add(vec_hash)
            unique_indices.append(i)
    
    if len(unique_indices) < len(candidate_vectors):
        removed = len(candidate_vectors) - len(unique_indices)
        print(f">>> Removed {removed} duplicate vectors across strategies")
        
        candidate_vectors    = [candidate_vectors[i] for i in unique_indices]
        candidate_labels_hot = [candidate_labels_hot[i] for i in unique_indices]
        candidate_labels_int = [candidate_labels_int[i] for i in unique_indices]
        candidate_sims       = [candidate_sims[i] for i in unique_indices]
        candidate_counts     = [candidate_counts[i] for i in unique_indices]
        candidate_sources    = [candidate_sources[i] for i in unique_indices]

print(f">>> Total merged candidates: {len(candidate_vectors)}")
if len(candidate_sources) > 0:
    from collections import Counter
    source_counts = Counter(candidate_sources)
    for src, cnt in source_counts.items():
        print(f"    - {src}: {cnt}")

# =========================================================
# PASS 2: ITERATIVE STM OPTIMIZATION
# =========================================================
print("\n_______________________________________________________________________")
print("PASS 2: Iterative STM Optimization (Label-Homogeneous Batches)")
print("_______________________________________________________________________")

if USING_STM and len(candidate_vectors) > 0:
    cand_vecs = np.array(candidate_vectors)
    cand_labels_hot = np.array(candidate_labels_hot)
    cand_labels_int = np.array(candidate_labels_int)
    cand_sims = np.array(candidate_sims)
    
    unique_labels = np.unique(cand_labels_int)
    homogeneous_batches = []
    
    print(f"Organizing {len(cand_vecs)} candidates into homogeneous batches...")
    
    for label in unique_labels:
        mask = (cand_labels_int == label)
        l_vecs = cand_vecs[mask]
        l_hots = cand_labels_hot[mask]
        l_ints = cand_labels_int[mask]
        l_sims = cand_sims[mask]
        
        l_sort_idx = np.argsort(l_sims)
        l_vecs = l_vecs[l_sort_idx]
        l_hots = l_hots[l_sort_idx]
        l_ints = l_ints[l_sort_idx]
        
        num_label_batches = int(np.ceil(len(l_vecs) / STM_INSERT_BATCH_SIZE))
        for i in range(num_label_batches):
            start = i * STM_INSERT_BATCH_SIZE
            end = min(start + STM_INSERT_BATCH_SIZE, len(l_vecs))
            
            batch_vecs = l_vecs[start:end]
            batch_hots = l_hots[start:end]
            batch_ints = l_ints[start:end]
            
            homogeneous_batches.append((batch_vecs, batch_hots, batch_ints))
    
    current_stm_vecs = []
    current_stm_labels = []
    
    baseline_acc = calculate_accuracy_with_stm(system_model, X_opt, y_opt_int, [], [])
    print(f"Baseline Accuracy (No STM): {baseline_acc:.4f}")
    
    best_acc = baseline_acc
    no_improve_count = 0
    total_inserted = 0
    
    start_time = time.time()
    
    for i, (batch_vecs, batch_labels_hot, batch_labels_int) in enumerate(homogeneous_batches):
        
        temp_stm_vecs = current_stm_vecs + [batch_vecs] if current_stm_vecs else [batch_vecs]
        temp_stm_vecs_np = np.vstack(temp_stm_vecs)
        temp_stm_labels_np = np.vstack(current_stm_labels + [batch_labels_hot]) if current_stm_labels else batch_labels_hot
        
        temp_acc = calculate_accuracy_with_stm(system_model, X_opt, y_opt_int, temp_stm_vecs_np, temp_stm_labels_np)
        
        if temp_acc > best_acc:
            best_acc = temp_acc
            current_stm_vecs.append(batch_vecs)
            current_stm_labels.append(batch_labels_hot)
            total_inserted += len(batch_vecs)
            no_improve_count = 0
            print(f"  Batch {i+1} (Class {batch_labels_int[0]}): ACCEPTED. New Acc: {best_acc:.4f}")
            
            ids_to_insert = [f"stm_opt_{total_inserted - len(batch_vecs) + j}" for j in range(len(batch_vecs))]
            metadatas_to_insert = []
            for idx in range(len(batch_vecs)):
                gt_vec = [0]*10
                gt_vec[int(batch_labels_int[idx])] = 1
                metadatas_to_insert.append({"true_label": int(batch_labels_int[idx]), "one_hot_vector": str(gt_vec)})
            
            stm_collection.add(
                embeddings=batch_vecs.tolist(),
                ids=ids_to_insert,
                metadatas=metadatas_to_insert
            )
            
        else:
            no_improve_count += 1
            print(f"  Batch {i+1} (Class {batch_labels_int[0]}): REJECTED.")
            
            if no_improve_count >= STM_PATIENCE:
                print(f"  >>> STM Patience Reached. Stopping Optimization.")
                break
    
    end_time = time.time()
    print(f"\n>>> Optimization Finished in {end_time - start_time:.2f} seconds.")
    print(f">>> Total Samples Inserted into STM: {total_inserted}")
    print(f">>> Final Optimized STM Accuracy (on Subset): {best_acc:.4f}")
    
    if len(current_stm_vecs) > 0:
        stm_vecs_final = np.vstack(current_stm_vecs)
        stm_labels_final = np.vstack(current_stm_labels)
    else:
        stm_vecs_final = np.empty((0, EMBEDDING_DIM))
        stm_labels_final = np.empty((0, 10))
else:
    stm_vecs_final = np.empty((0, EMBEDDING_DIM))
    stm_labels_final = np.empty((0, 10))
    print(">>> No STM Optimization Performed.")

# =========================================================
# PASS 3: FINAL EVALUATION WITH OPTIMIZED STM
# =========================================================
print("\n_______________________________________________________________________")
print("PASS 3: Final Evaluation on Full Test Set")
print("_______________________________________________________________________")

eval_dataset = tf.data.Dataset.from_tensor_slices((X_te, y_te_int, y_te_hot)).batch(BATCH_SIZE)

pass3_preds = []
pass3_trues = []

stm_v_tf = tf.constant(stm_vecs_final, dtype=tf.float32) if len(stm_vecs_final) > 0 else None
stm_l_tf = tf.constant(stm_labels_final, dtype=tf.float32) if len(stm_labels_final) > 0 else None

for step, (x_batch, y_true_int, y_true_hot) in enumerate(eval_dataset):
    output = system_model(x_batch, training=False, stm_vecs=stm_v_tf, stm_labels=stm_l_tf, return_sim=False)
    pred_final = output
    
    y_pred_cls = np.argmax(pred_final.numpy(), axis=1)
    y_true_cls = y_true_int.numpy()
    
    pass3_preds.extend(y_pred_cls)
    pass3_trues.extend(y_true_cls)

acc_pass3 = accuracy_score(pass3_trues, pass3_preds)
print(f">>> PASS 3 Final Accuracy: {acc_pass3:.4f}")

# =========================================================
# RECOVERY ANALYSIS (From Script A)
# =========================================================
print("\n_______________________________________________________________________")
print("RECOVERY ANALYSIS (Before vs After)")
print("_______________________________________________________________________")

pass1_preds = np.array(pass1_preds)
pass3_preds = np.array(pass3_preds)
pass1_trues = np.array(pass1_trues)

wrong_p1 = (pass1_preds != pass1_trues)
correct_p3 = (pass3_preds == pass1_trues)
recovered_mask = np.logical_and(wrong_p1, correct_p3)

total_wrong_p1 = np.sum(wrong_p1)
total_recovered = np.sum(recovered_mask)

print(f"Total Samples in Test Set: {len(pass1_trues)}")
print(f"Wrong in Pass 1: {total_wrong_p1}")
print(f"Recovered in Pass 3: {total_recovered}")

if total_wrong_p1 > 0:
    recovery_rate = total_recovered / total_wrong_p1
    print(f"Recovery Rate: {recovery_rate:.2%} of previous errors fixed by Optimized STM")

# Final Comparison
print("\nRunning Baseline Value Encoder...")
y_ve_logits = system_model.value_encoder.predict(X_te, verbose=0)
y_ve_cls = np.argmax(y_ve_logits, axis=1)
acc_ve = accuracy_score(pass1_trues, y_ve_cls)

print(f"\nFinal Accuracy Comparison:")
print(f"Value Encoder (Baseline)      : {acc_ve:.4f}")
print(f"Multi-Hop Hyper Pass 1 (No STM): {acc_pass1:.4f}")
print(f"Multi-Hop Hyper Pass 3 (Opt STM): {acc_pass3:.4f}")

if acc_pass3 > acc_pass1:
    print(f"\n*** OPTIMIZED STM IMPROVEMENT: +{(acc_pass3 - acc_pass1)*100:.2f}% ***")
else:
    print(f"\n*** STM did not improve overall accuracy (Possible Overfitting on Subset) ***")

print("\nClassification Report (Pass 3):")
print(classification_report(pass3_trues, pass3_preds))

# ---------------------------------------------------------
# 11. SAVE MODEL & EDA DATA
# ---------------------------------------------------------
print("\nSaving Trained Multi-Hop Hyper Retriever...")
tf.saved_model.save(retriever_branch, SAVE_PATH_HQE_SYSTEM)
print(f"Saved to: {SAVE_PATH_HQE_SYSTEM}")

if ENABLE_CONSOLIDATION_EDA and eda_queries:
    print("\nSaving EDA Manifold Snapshots...")
    os.makedirs(EDA_SAVE_PATH, exist_ok=True)
    with open(os.path.join(EDA_SAVE_PATH, "query_evolution.pkl"), "wb") as f:
        pickle.dump(eda_queries, f)
    print(f"Saved EDA data to: {EDA_SAVE_PATH}")

print("\n_______________________________________________________________________")
print("Training Complete!")
print("_______________________________________________________________________")