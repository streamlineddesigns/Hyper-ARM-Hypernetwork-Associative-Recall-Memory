# ---------------------------------------------------------
# HYPER QUERY ENCODER SCRIPT (Continuous Learning)
# Multi-Hop + Hypernetwork Per Hop + Learnable Temperature + STM
# LTM & STM Persist Across Runs (FIFO Eviction - Continuous)
# Model Weights Loading from Previous Training (Unified .keras)
# Visual Centroids Persisted Across Runs
# HQE Encoder for LTM on Subsequent Runs
# No Duplicate ID Warnings
# STM Database Vacuum/Cleanup
# Learning Rate Persistence (Optimizer State Saved)
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
import sqlite3
import warnings
from datetime import datetime
import json

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
STM_SIMILARITY_THRESHOLD_CAND = 0.75
STM_SIMILARITY_THRESHOLD_KEEP = 1.0

# Model Paths (Unified Weights Only)
ENCODER_PATH = "./saved_cnne_model_dir"
VALUE_ENC_PATH = "./saved_mnist_classifier_dir" 
# *** UPDATED: Single .keras weights file + Full model ***
SAVE_PATH_HQE_WEIGHTS = "./saved_hqe_hyper_multi_hop_weights.keras"
SAVE_PATH_HQE_FULL = "./saved_hqe_hyper_multi_hop_full.keras"
SAVE_PATH_HQE_CONFIG = "./saved_hqe_hyper_multi_hop_config.json"

# *** NEW: Visual Centroids Path ***
SAVE_PATH_CENTROIDS = "./saved_visual_centroids.npy"

# Embedding & Architecture (Script A dims + Script B Hypernetwork)
EMBEDDING_DIM = 128 
NUM_NEIGHBORS = 5       
BATCH_SIZE = 128
EPOCHS = 5    
LEARNING_RATE = 0.0003

# Multi-Hop Configuration (From Script A)
NUM_HOPS = 1

# Temperature Config (From Script A)
MIN_TEMP = 0.5
MAX_TEMP = 2.0
INIT_TEMP = 1.0 

# STM Optimization Config (From Script A)
STM_INSERT_BATCH_SIZE = 256
STM_OPTIMIZATION_SUBSET_RATIO = 0.5
STM_LTM_MIX_OPTIMIZATION_RATIO = 0.2
STM_PATIENCE = 16
STM_BOOTSTRAP_TOTAL = 0  # 10 batches * 32 samples

# LTM Optimization
LTM_INSERT_BATCH_SIZE = 256
LTM_OPTIMIZATION_SUBSET_RATIO = 0.5
LTM_SIMILARITY_THRESHOLD_CAND = 0.75  # *** INCREASED for Run 2+ ***
LTM_SIMILARITY_THRESHOLD_KEEP = 1.0  # *** INCREASED for Run 2+ ***
LTM_PATIENCE = 8

# *** NEW: Capacity Limits ***
LTM_MAX_CAPACITY = 8192
STM_MAX_CAPACITY = 4096

# STM Candidate Retrieval Config (From Script A)
STM_LTM_MIN_SIM = 0.6
STM_DEDUP_CANDIDATES = True
STM_SORT_MODE = "similarity"

# STM Storage Config (Q vs Z)
STM_STORE_Q_NOT_Z_STRAT1 = False   # Strategy 1: Store Q (multi-hop transformed) instead of Z (frozen encoder)
STM_STORE_Q_NOT_Z_STRAT2 = False  # Strategy 2: Keep LTM prototypes as Z (consistent with LTM storage)
NEGATIVE_AVOIDANCE_WEIGHT = 1.0

# Hybrid Strategy Flags (From Script A)
HYBRID_USE_LOW_SIM = True
HYBRID_USE_LTM_PROTO = True

# Hypernetwork Config (From Script B)
NUM_VISUAL_CENTROIDS = 10
NUM_ACTIONS = 10
TARGET_NET_ARCH = [64, 32]
HYPER_INTERMEDIATE_DIM = 98

# EDA Config (From Script B)
ENABLE_CONSOLIDATION_EDA = False
EDA_SAVE_PATH = "./eda_manifold_snapshots"

# *** NEW: Persistence Flags ***
PERSIST_LTM_ACROSS_RUNS = True
PERSIST_STM_ACROSS_RUNS = True
PERSIST_CENTROIDS_ACROSS_RUNS = True
LOAD_PREVIOUS_MODEL = True

# *** NEW: Centroid Usage Flag ***
USE_EXISTING_CENTROIDS_DIRECTLY = True  # True = Skip K-Means, Load as-is
                                         # False = Use as K-Means seeds (refine)

# *** NEW: LTM Encoding Strategy Flags (Ablation Study) ***
LTM_USE_FROZEN_ENCODER_FOR_INSERTION = True     # True = Z for insert
                                                # False = Q for insert
LTM_USE_HQE_FOR_RETRIEVAL = True                # True = Q for search
                                                # False = Z for search

# *** NEW: Dynamic LTM/STM Weighting ***
DYNAMIC_WEIGHTING = True
LTM_CONFIDENCE_THRESHOLD = 0.25     # Below this = increase STM weight
STM_BASE_WEIGHT = 0.3
STM_MAX_WEIGHT = 0.75
STM_MIN_WEIGHT = 0.0
WEIGHT_BOOST_FACTOR = 1.5 #1.0=Linear, 1.25=EaseInOutSine, 1.5=EaseInOutQuad, 2.0=EasInOutExpo (Linear approximations)
LOG_CONFIDENCE_SCORES = False

# Add this after your CONFIGURATION section:
GLOBAL_STM_VECS = None
GLOBAL_STM_LABELS = None

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
    """
    Encode images using either a Keras model or SavedModel encoder.
    Detects model type automatically.
    """
    outs = []
    n = len(images)
    
    # *** DETECT MODEL TYPE ***
    is_keras_model = isinstance(module, keras.Model) or hasattr(module, 'predict')
    
    for i in range(0, n, batch_size):
        batch = images[i:i + batch_size]
        
        if is_keras_model:
            # *** Keras Model: Call directly with encode_only=True ***
            z = module(batch, training=False, encode_only=True)
            if isinstance(z, dict):
                z = list(z.values())[0]
            elif isinstance(z, (list, tuple)):
                z = z[0]
        else:
            # *** SavedModel: Use frozen encoder caller ***
            z = call_frozen_encoder(module, batch)
        
        z = tf.reshape(z, [tf.shape(z)[0], -1])
        outs.append(z.numpy())
    
    return np.concatenate(outs, axis=0)

# ---------------------------------------------------------
# *** FIXED: Helper Functions for FIFO Eviction ***
# ---------------------------------------------------------
def get_collection_count(collection):
    """Get current number of vectors in collection"""
    try:
        results = collection.get(include=[])
        return len(results['ids'])
    except:
        return 0

def get_oldest_ids(collection, num_to_remove):
    """Get IDs of oldest vectors based on insertion timestamp"""
    try:
        results = collection.get(
            include=['metadatas'],
            limit=num_to_remove
        )
        if len(results['ids']) == 0:
            return []
        
        # Sort by timestamp in metadata
        ids_with_time = []
        for i, meta in enumerate(results['metadatas']):
            ts = meta.get('insert_timestamp', 0)
            ids_with_time.append((results['ids'][i], ts))
        
        ids_with_time.sort(key=lambda x: x[1])
        return [x[0] for x in ids_with_time[:num_to_remove]]
    except Exception as e:
        print(f"    Warning: Could not get oldest IDs: {e}")
        try:
            results = collection.get(include=[], limit=num_to_remove)
            return results['ids'][:num_to_remove]
        except:
            return []

def make_room_for_insert(collection, num_to_add, max_capacity, collection_name):
    """
    *** FIXED: Make room for new vectors by evicting oldest (FIFO) ***
    Returns number of vectors evicted
    """
    current_count = get_collection_count(collection)
    num_evicted = 0
    
    if current_count + num_to_add > max_capacity:
        num_to_remove = (current_count + num_to_add) - max_capacity
        oldest_ids = get_oldest_ids(collection, num_to_remove)
        if len(oldest_ids) > 0:
            collection.delete(ids=oldest_ids)
            num_evicted = len(oldest_ids)
            print(f"    [{collection_name}] Evicted {num_evicted} oldest vectors (FIFO) to make room")
    
    return num_evicted

# ---------------------------------------------------------
# *** NEW: Vacuum SQLite Database After Deletions ***
# ---------------------------------------------------------
def vacuum_chroma_database(db_path):
    """
    Vacuum the ChromaDB SQLite database to reclaim space after deletions
    """
    try:
        db_file = os.path.join(db_path, "chroma.sqlite3")
        if os.path.exists(db_file):
            conn = sqlite3.connect(db_file)
            conn.execute("VACUUM")
            conn.close()
            print(f"    [DB] Vacuumed {db_file}")
            return True
    except Exception as e:
        print(f"    [DB] Warning: Could not vacuum database: {e}")
    return False

# ---------------------------------------------------------
# *** NEW: Generate Unique IDs with No Duplicates ***
# ---------------------------------------------------------
def generate_unique_id(prefix, counter, timestamp):
    """Generate unique ID that won't conflict with existing IDs"""
    return f"{prefix}_{timestamp}_{counter}"

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
# 2. Load Frozen Encoder (Needed for Centroids + LTM Seeding)
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("Loading Frozen Encoder")
print("_______________________________________________________________________")
print(f"Loading Frozen Encoder from {ENCODER_PATH}...")
loaded_encoder = tf.saved_model.load(ENCODER_PATH)
print("Frozen Encoder loaded successfully.")

# ---------------------------------------------------------
# 3. Visual Centroids (MOVED UP - BEFORE HQE Model Build)
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("Generating Visual Centroids for Hypernetwork Context")
print("_______________________________________________________________________")

# Generate latent vectors for training data
print("Encoding training data for K-Means...")
Z_train_val = encode_images(loaded_encoder, X_train_val, batch_size=256)
print(f"Z_train shape: {Z_train_val.shape}")

# *** Check if existing centroids exist ***
CENTROIDS_EXIST = False
EXISTING_CENTROIDS = None

if PERSIST_CENTROIDS_ACROSS_RUNS and os.path.exists(SAVE_PATH_CENTROIDS):
    print(f"Previous centroids found at {SAVE_PATH_CENTROIDS}")
    try:
        EXISTING_CENTROIDS = np.load(SAVE_PATH_CENTROIDS)
        
        # Validate shape
        if EXISTING_CENTROIDS.shape == (NUM_VISUAL_CENTROIDS, EMBEDDING_DIM):
            CENTROIDS_EXIST = True
            print(f"Loaded {NUM_VISUAL_CENTROIDS} existing centroids")
            print(f"  Shape: {EXISTING_CENTROIDS.shape}")
        else:
            print(f"Warning: Centroid shape mismatch. Expected ({NUM_VISUAL_CENTROIDS}, {EMBEDDING_DIM}), got {EXISTING_CENTROIDS.shape}")
            EXISTING_CENTROIDS = None
    except Exception as e:
        print(f"Failed to load centroids: {e}")
        EXISTING_CENTROIDS = None
else:
    print("No previous centroids found. Will use random initialization.")

# *** Decide Strategy Based on Flag ***
if USE_EXISTING_CENTROIDS_DIRECTLY and CENTROIDS_EXIST:
    # === STRATEGY 1: Direct Load (No K-Means) ===
    print("\n>>> Using EXISTING CENTROIDS Directly (No K-Means Refinement)")
    VISUAL_CENTROIDS = EXISTING_CENTROIDS.astype('float32')
    KMEANS_RAN = False
else:
    # === STRATEGY 2: K-Means (With or Without Seeds) ===
    print(f"\nRunning K-Means with K={NUM_VISUAL_CENTROIDS}...")
    
    if CENTROIDS_EXIST and not USE_EXISTING_CENTROIDS_DIRECTLY:
        # Use existing as seeds
        kmeans = KMeans(
            n_clusters=NUM_VISUAL_CENTROIDS, 
            init=EXISTING_CENTROIDS,
            n_init=1,
            random_state=42,
            max_iter=300
        )
        print("  Initialization: Seeded with existing centroids (Refinement Mode)")
    else:
        # Random initialization
        kmeans = KMeans(
            n_clusters=NUM_VISUAL_CENTROIDS, 
            init='k-means++',
            n_init=10,
            random_state=42,
            max_iter=300
        )
        print("  Initialization: Random (k-means++)")
    
    kmeans.fit(Z_train_val)
    VISUAL_CENTROIDS = kmeans.cluster_centers_.astype('float32')
    KMEANS_RAN = True
    print(f"Generated {NUM_VISUAL_CENTROIDS} visual centroids.")

# *** Save centroids for future runs ***
print(f"Saving centroids to {SAVE_PATH_CENTROIDS}...")
np.save(SAVE_PATH_CENTROIDS, VISUAL_CENTROIDS)
print("Centroids saved successfully.")

# Convert to TensorFlow constant for model use
CENTROID_VECS = tf.constant(VISUAL_CENTROIDS)
print(f"Visual Centroids ready: {CENTROID_VECS.shape}")
print(f"K-Means Execution: {'SKIPPED' if not KMEANS_RAN else 'EXECUTED'}")

# ---------------------------------------------------------
# 4. Build HQE Model (MOVED UP - BEFORE LTM Seeding)
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("Model Initialization (Before LTM Seeding)")
print("_______________________________________________________________________")

# Model Architecture Classes
class FrozenEncoderLayer(layers.Layer):
    """Robust Frozen Encoder Layer (From Script B)"""
    def __init__(self, module, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self.trainable = False
        
    def call(self, inputs, training=False):
        res = call_frozen_encoder(self.module, inputs)
        return tf.reshape(res, [tf.shape(res)[0], -1]) 
    
    def get_config(self):
        # Cannot serialize SavedModel encoder - will be passed at runtime
        base_config = super().get_config()
        return {**base_config, 'module': 'saved_model_encoder'}


class ResidualCNN(keras.Model):
    """Residual CNN per hop (From Script B)"""
    def __init__(self, target_dim, hop_id=0):
        super().__init__()
        self.hop_id = hop_id
        self.target_dim = target_dim
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
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'target_dim': self.target_dim, 'hop_id': self.hop_id}
    
    @classmethod
    def from_config(cls, config):
        return cls(
            target_dim=config.get('target_dim', 128),
            hop_id=config.get('hop_id', 0)
        )


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
        self.output_param_count = output_param_count
        self.net = keras.Sequential([
            layers.Dense(self.intermediate_dim, activation='relu', name=f"hyper_hop{hop_id}_dense1"),
            layers.Dense(output_param_count, activation='linear', name=f"hyper_hop{hop_id}_dense2")
        ])
    
    def call(self, centroid_context):
        return self.net(centroid_context)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'output_param_count': self.output_param_count, 'hop_id': self.hop_id, 'intermediate_dim': self.intermediate_dim}
    
    @classmethod
    def from_config(cls, config):
        return cls(
            output_param_count=config.get('output_param_count', 98),
            hop_id=config.get('hop_id', 0)
        )


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
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'arch_list': self.arch_list, 'output_dim': self.output_dim, 'hop_id': self.hop_id}
    
    @classmethod
    def from_config(cls, config):
        return cls(
            arch_list=config.get('arch_list', [64, 32]),
            output_dim=config.get('output_dim', 128),
            hop_id=config.get('hop_id', 0)
        )


class MultiHopHyperRetriever(Model):
    """
    Multi-Hop with 1:1 CNN + Hypernetwork Per Hop
    Retrieval uses Direct Cosine Similarity (From Script A)
    Learnable Temperature (From Script A)
    """
    def __init__(self, enc, num_hops, target_dim, hyper_arch, output_dim, 
                 initial_temperature=1.0, saved_learning_rate=None):
        super().__init__()
        self.enc = enc
        self.num_hops = num_hops
        self.target_dim = target_dim
        self.hyper_arch = hyper_arch
        self.output_dim = output_dim
        self.initial_temperature = initial_temperature
        self.saved_learning_rate = saved_learning_rate
        self._encoder_set = enc is not None
        
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
    
    # =========================================================
    # === SERIALIZATION METHODS - FIXED FOR KERAS LOAD ===
    # =========================================================
    def get_config(self):
        """
        Serialize model configuration for saving.
        NOTE: 'enc' cannot be serialized - we store a placeholder.
        """
        base_config = super().get_config()
        return {
            **base_config,
            'num_hops': self.num_hops,
            'target_dim': self.target_dim,
            'hyper_arch': self.hyper_arch,
            'output_dim': self.output_dim,
            'initial_temperature': self.initial_temperature,
            'saved_learning_rate': self.saved_learning_rate,
        }
    
    @classmethod
    def from_config(cls, config):
        """
        Reconstruct model from config.
        NOTE: Encoder will be None - must be replaced after loading!
        Keras calls this automatically during load_model().
        """
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        
        # Set attributes from config
        instance.num_hops = config.get('num_hops', 1)
        instance.target_dim = config.get('target_dim', 128)
        instance.hyper_arch = config.get('hyper_arch', [64, 32])
        instance.output_dim = config.get('output_dim', 128)
        instance.initial_temperature = config.get('initial_temperature', 1.0)
        instance.saved_learning_rate = config.get('saved_learning_rate', None)
        instance.enc = None  # Will be replaced after loading
        instance._encoder_set = False  # Track encoder status
        
        # Initialize Model base class
        super(MultiHopHyperRetriever, instance).__init__()
        
        # Create layers (will have random weights initially)
        instance.hop_cnns = [ResidualCNN(target_dim=instance.target_dim, hop_id=i) 
                            for i in range(instance.num_hops)]
        instance.hop_hypernets = [CentroidHypernetwork(
            output_param_count=get_target_params_count(instance.target_dim, instance.hyper_arch, instance.output_dim),
            hop_id=i
        ) for i in range(instance.num_hops)]
        instance.hop_target_nets = [DynamicTargetNetwork(
            arch_list=instance.hyper_arch,
            output_dim=instance.output_dim,
            hop_id=i
        ) for i in range(instance.num_hops)]
        
        # Create temperature variable
        instance.log_temp = tf.Variable(
            np.log(instance.initial_temperature), 
            trainable=True, 
            dtype=tf.float32, 
            name="learnable_log_temperature"
        )
        
        return instance
    # =========================================================
        
    def get_temperature(self):
        temp = tf.exp(self.log_temp)
        return tf.clip_by_value(temp, MIN_TEMP, MAX_TEMP)
        
    def call(self, inputs, training=None, stm_vecs=None, stm_labels=None, 
            return_sim=False, return_intermediate=False, encode_only=False):
        # === STEP 1: Base Encoding ===
        # REMOVED: Don't check encoder during load - Keras may call internally
        # Just use dummy encoding if encoder is None (during deserialization)
        if self.enc is not None:
            z_base = self.enc(inputs, training=training)
        else:
            # During load/serialization, create dummy encoding
            z_base = tf.zeros((tf.shape(inputs)[0], self.target_dim), dtype=tf.float32)
        
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
        
        # === ENCODE ONLY MODE: Skip retrieval entirely ===
        if encode_only:
            if return_intermediate:
                return final_q, intermediate_queries, hop_data
            return final_q
        
        # Add noise during training (From Script A)
        if training:
            noise = tf.random.normal(shape=tf.shape(final_q), mean=0.0, stddev=0.01)
            final_q = final_q + noise
        
        # === STEP 3: Direct Cosine Similarity Retrieval (From Script A) ===
        # Note: MEM_BANK_VECS will be set during LTM Initialization
        main_vecs_norm = tf.nn.l2_normalize(MEM_BANK_VECS, axis=1)
        sim_matrix_main = tf.matmul(final_q, main_vecs_norm, transpose_b=True)
        values_main, indices_main = tf.math.top_k(sim_matrix_main, k=NUM_NEIGHBORS)  
        max_sim_main = tf.reduce_max(values_main, axis=1)#max un-normalized sim              
        current_temp = self.get_temperature()
        scaled_values_main = values_main / current_temp 
        attn_weights_main = tf.nn.softmax(scaled_values_main, axis=1)
        neighbor_labels_main = tf.gather(MEM_BANK_LABELS, indices_main) 
        pred_main = tf.reduce_sum(tf.expand_dims(attn_weights_main, -1) * neighbor_labels_main, axis=1)
        
        pred_final = pred_main

        #training only
        if GLOBAL_STM_VECS is not None:
            stm_vecs = GLOBAL_STM_VECS
            stm_labels = GLOBAL_STM_LABELS

        # === STEP 4: STM Retrieval (From Script A) ===
        if stm_vecs is not None and tf.shape(stm_vecs)[0] > 0:
            stm_vecs_norm = tf.nn.l2_normalize(stm_vecs, axis=1)
            sim_matrix_stm = tf.matmul(final_q, stm_vecs_norm, transpose_b=True)
            k_stm = tf.minimum(NUM_NEIGHBORS, tf.shape(stm_vecs_norm)[0])
            values_stm, indices_stm = tf.math.top_k(sim_matrix_stm, k=k_stm)    
            max_sim_stm = tf.reduce_max(values_stm, axis=1)#max un-normalized sim        
            scaled_values_stm = values_stm / current_temp 
            attn_weights_stm = tf.nn.softmax(scaled_values_stm, axis=1)
            
            neighbor_labels_stm = tf.gather(stm_labels, indices_stm) 
            pred_stm = tf.reduce_sum(tf.expand_dims(attn_weights_stm, -1) * neighbor_labels_stm, axis=1)

            ltm_confidence = max_sim_main  # Shape: [batch_size]
            stm_confidence = max_sim_stm  # Shape: [batch_size]

            confidence_is_below_threshold = ltm_confidence[0] <= LTM_CONFIDENCE_THRESHOLD
                
            # === DYNAMIC WEIGHTING BASED ON LTM CONFIDENCE ===
            if (confidence_is_below_threshold):
                # 1. Use tf.stack with axis=-1 to go from two [batch_size] -> [batch_size, 2]
                unnormalized_confidence_values = tf.stack((ltm_confidence, stm_confidence), axis=-1)
                # 2. Softmax across axis=-1 normalizes the values between LTM and STM per sample
                confidence_attention_weights = tf.nn.softmax(unnormalized_confidence_values, axis=-1)
                # 3. Extract weights using slice indexing to preserve dimensions for broadcasting
                # confidence_attention_weights is [batch_size, 2]. Slicing [:, 0] gives [batch_size]. 
                # Adding tf.newaxis expands it cleanly to [batch_size, 1].
                ltm_w = confidence_attention_weights[:, 0, tf.newaxis]
                stm_w = confidence_attention_weights[:, 1, tf.newaxis]

                # Boost STM weight based on deficit
                ltm_confidence_deficit = tf.maximum(0.0, LTM_CONFIDENCE_THRESHOLD - ltm_w)
                stm_weight = (ltm_confidence_deficit * WEIGHT_BOOST_FACTOR)
                stm_weight = tf.clip_by_value(stm_weight, STM_MIN_WEIGHT, STM_MAX_WEIGHT)
                ltm_weight = 1.0 - stm_weight

                # 4. Compute the final prediction safely
                pred_final = (pred_main * ltm_weight + pred_stm * stm_weight)
                
                # Optional: Log weighting for debugging
                if LOG_CONFIDENCE_SCORES:
                    tf.print(f"  [Conf] LTM={tf.reduce_mean(max_sim_main):.3f} STM={tf.reduce_mean(max_sim_stm):.3f}", summarize=-1) 

        
        if return_intermediate:
            if return_sim:
                return {'predictions': pred_final, 'max_similarity': max_sim_main}, intermediate_queries, final_q, hop_data
            return pred_final, intermediate_queries, final_q, hop_data
        else:
            if return_sim:
                return {'predictions': pred_final, 'max_similarity': max_sim_main}
            return pred_final
    
    def set_encoder(self, encoder_layer):
        """Set encoder layer after loading model"""
        self.enc = encoder_layer
        self._encoder_set = True
        print(f"  ✓ Encoder set: {type(encoder_layer).__name__}")
    
    def check_encoder(self):
        """Check if encoder is properly set"""
        return self._encoder_set and self.enc is not None

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

#Custom Objects Registration
CUSTOM_OBJECTS = {
    'FrozenEncoderLayer': FrozenEncoderLayer,
    'ResidualCNN': ResidualCNN,
    'CentroidHypernetwork': CentroidHypernetwork,
    'DynamicTargetNetwork': DynamicTargetNetwork,
    'MultiHopHyperRetriever': MultiHopHyperRetriever,
    'GuidedSystem': GuidedSystem,
}

keras.utils.get_custom_objects().update(CUSTOM_OBJECTS)

# ---------------------------------------------------------
# VERIFICATION: Check Model Loaded Correctly
# ---------------------------------------------------------
def verify_model_loading(model, model_name="HQE"):
    """
    Comprehensive verification that model loaded correctly
    Catches silent failures before wasting training time
    """
    print(f"\n{'='*60}")
    print(f"MODEL LOADING VERIFICATION: {model_name}")
    print(f"{'='*60}")
    
    verification_passed = True
    issues_found = []
    
    # 1. Check Model Type
    print("\n[1] Model Type Check:")
    expected_type = MultiHopHyperRetriever
    actual_type = type(model)
    if isinstance(model, expected_type):
        print(f"  ✓ Model type correct: {actual_type.__name__}")
    else:
        print(f"  ✗ Model type mismatch!")
        print(f"    Expected: {expected_type.__name__}")
        print(f"    Got: {actual_type.__name__}")
        verification_passed = False
        issues_found.append("Model type mismatch")
    
    # 2. Check Layer Count & Names
    print("\n[2] Layer Structure Check:")
    expected_layers = {
        'FrozenEncoderLayer': 1,
        'ResidualCNN': NUM_HOPS,
        'CentroidHypernetwork': NUM_HOPS,
        'DynamicTargetNetwork': NUM_HOPS,
    }
    
    layer_counts = {}
    for layer in model.layers:
        layer_type = type(layer).__name__
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        print(f"  - {layer_type}: {layer.name}")
    
    for layer_type, expected_count in expected_layers.items():
        actual_count = layer_counts.get(layer_type, 0)
        if actual_count == expected_count:
            print(f"  ✓ {layer_type}: {actual_count}/{expected_count}")
        else:
            print(f"  ✗ {layer_type}: {actual_count}/{expected_count} (MISMATCH!)")
            verification_passed = False
            issues_found.append(f"{layer_type} count mismatch")
    
    # 3. Check Trainable Variables
    print("\n[3] Trainable Variables Check:")
    trainable_vars = model.trainable_variables
    print(f"  Total trainable variables: {len(trainable_vars)}")
    
    if len(trainable_vars) > 0:
        print(f"  ✓ Model has trainable parameters")
        
        # Check temperature variable exists
        temp_var_found = False
        for var in trainable_vars:
            if 'temperature' in var.name.lower() or 'log_temp' in var.name.lower():
                temp_var_found = True
                current_temp = model.get_temperature().numpy()
                print(f"  ✓ Temperature variable found: {current_temp:.4f}")
                break
        
        if not temp_var_found:
            print(f"  ✗ Temperature variable NOT found!")
            verification_passed = False
            issues_found.append("Temperature variable missing")
    else:
        print(f"  ✗ No trainable variables found!")
        verification_passed = False
        issues_found.append("No trainable variables")
    
    # 4. Check Weight Values (Not All Zeros)
    print("\n[4] Weight Values Check:")
    zero_weight_layers = 0
    total_layers = 0
    nan_weight_layers = 0
    
    for layer in model.layers:
        if hasattr(layer, 'weights') and len(layer.weights) > 0:
            total_layers += 1
            for w in layer.weights:
                w_numpy = w.numpy()
                if np.all(w_numpy == 0):
                    zero_weight_layers += 1
                    break
                if np.any(np.isnan(w_numpy)):
                    nan_weight_layers += 1
                    break
    
    if zero_weight_layers == 0:
        print(f"  ✓ All {total_layers} layers have non-zero weights")
    else:
        print(f"  ✗ {zero_weight_layers}/{total_layers} layers have all-zero weights!")
        verification_passed = False
        issues_found.append(f"{zero_weight_layers} layers with zero weights")
    
    if nan_weight_layers == 0:
        print(f"  ✓ No NaN weights detected")
    else:
        print(f"  ✗ {nan_weight_layers} layers have NaN weights!")
        verification_passed = False
        issues_found.append(f"{nan_weight_layers} layers with NaN weights")
    
    # 5. Forward Pass Test
    print("\n[5] Forward Pass Test:")
    try:
        test_input = tf.random.normal((2, 28, 28, 1), dtype=tf.float32)
        output = model(test_input, training=False)
        
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        if output.shape == (2, NUM_ACTIONS):
            print(f"  ✓ Output shape correct")
        else:
            print(f"  ✗ Output shape incorrect! Expected (2, {NUM_ACTIONS})")
            verification_passed = False
            issues_found.append(f"Output shape mismatch: {output.shape}")
        
        # Check output is not all zeros/nans
        if np.any(np.isnan(output.numpy())):
            print(f"  ✗ Output contains NaN values!")
            verification_passed = False
            issues_found.append("NaN in output")
        elif np.all(output.numpy() == 0):
            print(f"  ✗ Output is all zeros!")
            #verification_passed = False
            #issues_found.append("All-zero output")
        else:
            print(f"  ✓ Output values valid (min={output.numpy().min():.4f}, max={output.numpy().max():.4f})")
            
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        verification_passed = False
        issues_found.append(f"Forward pass exception: {str(e)}")
    
    # 6. Check Custom Layers Have Correct Attributes
    print("\n[6] Custom Layer Attributes Check:")
    
    # Check hypernetworks have intermediate_dim
    for i, layer in enumerate(model.layers):
        if isinstance(layer, CentroidHypernetwork):
            if hasattr(layer, 'intermediate_dim'):
                print(f"  ✓ Hypernetwork {i} has intermediate_dim: {layer.intermediate_dim}")
            else:
                print(f"  ✗ Hypernetwork {i} missing intermediate_dim!")
                verification_passed = False
                issues_found.append(f"Hypernetwork {i} missing intermediate_dim")
    
    # Check ResidualCNN has hop_id
    for i, layer in enumerate(model.layers):
        if isinstance(layer, ResidualCNN):
            if hasattr(layer, 'hop_id'):
                print(f"  ✓ ResidualCNN {i} has hop_id: {layer.hop_id}")
            else:
                print(f"  ✗ ResidualCNN {i} missing hop_id!")
                verification_passed = False
                issues_found.append(f"ResidualCNN {i} missing hop_id")
    
    # 7. Check Encoder Layer
    print("\n[7] Encoder Layer Check:")
    if hasattr(model, 'enc') and model.enc is not None:
        print(f"  ✓ Encoder layer exists: {type(model.enc).__name__}")
        try:
            test_input = tf.random.normal((1, 28, 28, 1), dtype=tf.float32)
            test_enc_output = model.enc(test_input, training=False)
            print(f"  ✓ Encoder forward pass works: {test_enc_output.shape}")
        except Exception as e:
            print(f"  ✗ Encoder forward pass failed: {e}")
            verification_passed = False
            issues_found.append(f"Encoder forward pass failed")
    else:
        print(f"  ✗ Encoder layer missing or None!")
        verification_passed = False
        issues_found.append("Encoder layer missing")
    
    # 8. Check Multi-Hop Structure
    print("\n[8] Multi-Hop Structure Check:")
    if hasattr(model, 'hop_cnns') and len(model.hop_cnns) == NUM_HOPS:
        print(f"  ✓ hop_cnns: {len(model.hop_cnns)} layers")
    else:
        print(f"  ✗ hop_cnns mismatch!")
        verification_passed = False
        issues_found.append("hop_cnns mismatch")
    
    if hasattr(model, 'hop_hypernets') and len(model.hop_hypernets) == NUM_HOPS:
        print(f"  ✓ hop_hypernets: {len(model.hop_hypernets)} layers")
    else:
        print(f"  ✗ hop_hypernets mismatch!")
        verification_passed = False
        issues_found.append("hop_hypernets mismatch")
    
    if hasattr(model, 'hop_target_nets') and len(model.hop_target_nets) == NUM_HOPS:
        print(f"  ✓ hop_target_nets: {len(model.hop_target_nets)} layers")
    else:
        print(f"  ✗ hop_target_nets mismatch!")
        verification_passed = False
        issues_found.append("hop_target_nets mismatch")
    
    # FINAL RESULT
    print(f"\n{'='*60}")
    if verification_passed:
        print("✓✓✓ ALL VERIFICATION CHECKS PASSED ✓✓✓")
        print(f"{'='*60}\n")
        return True
    else:
        print("✗✗✗ SOME VERIFICATION CHECKS FAILED ✗✗✗")
        print(f"\nIssues Found ({len(issues_found)}):")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        print(f"\n{'='*60}\n")
        return False


# ---------------------------------------------------------
# CUSTOM SAVE/LOAD FUNCTIONS FOR HQE MODEL
# ---------------------------------------------------------

def save_hqe_model(model, optimizer, filepath, save_weights_backup=True):
    """
    Save HQE model WITH optimizer state (includes learning rate)
    Returns tuple of (config_path, weights_path, optimizer_path)
    """
    print(f"\nSaving HQE model to {filepath}...")
    
    # 1. Save architecture config (WITH learning rate)
    config = {
        'num_hops': model.num_hops,
        'target_dim': model.target_dim,
        'hyper_arch': model.hyper_arch,
        'output_dim': model.output_dim,
        'initial_temperature': float(model.initial_temperature),
        'learning_rate': float(optimizer.learning_rate.numpy()),
        'optimizer_type': type(optimizer).__name__,
    }
    
    config_path = filepath.replace('_full.keras', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Config saved to {config_path}")
    print(f"  ✓ Learning rate saved: {config['learning_rate']:.9f}")
    
    # 2. Save model weights
    weights_path = filepath.replace('_full.keras', '_weights.keras')
    model.save_weights(weights_path)
    print(f"  ✓ Weights saved to {weights_path}")
    
    # 3. Save optimizer state separately
    optimizer_path = filepath.replace('_full.keras', '_optimizer.keras')
    optimizer.save_weights(optimizer_path)
    print(f"  ✓ Optimizer state saved to {optimizer_path}")
    
    # 4. Also save as full .keras (may have limitations with encoder)
    try:
        model.save(filepath)
        print(f"  ✓ Full model saved to {filepath}")
    except Exception as e:
        print(f"  ⚠ Full model save failed: {e}")
        print(f"  → Use config + weights files instead")
    
    # 5. Save backup weights file
    if save_weights_backup:
        weights_backup_path = filepath.replace('_full.keras', '_weights.keras')
        model.save_weights(weights_backup_path)
        print(f"  ✓ Weights backup saved to {weights_backup_path}")
    
    return config_path, weights_path, optimizer_path


def load_hqe_model(filepath, encoder_layer, custom_objects=None):
    """
    Load HQE model WITH optimizer state (includes learning rate)
    Returns: (model, optimizer, learning_rate, success_bool)
    """
    print(f"\nLoading HQE model from {filepath}...")
    
    if custom_objects is None:
        custom_objects = CUSTOM_OBJECTS
    
    loaded_lr = LEARNING_RATE  # Default fallback
    loaded_optimizer = None
    
    # Strategy 1: Try loading full .keras model
    if os.path.exists(filepath):
        try:
            print("  Trying full .keras load...")
            loaded_model = keras.models.load_model(
                filepath,
                custom_objects=custom_objects,
                compile=False  # Don't compile - avoids call() triggers
            )
            
            if isinstance(loaded_model, MultiHopHyperRetriever):
                # Replace encoder layer IMMEDIATELY (before any operations)
                loaded_model.set_encoder(encoder_layer)
                
                # Load config to get learning rate
                config_path = filepath.replace('_full.keras', '_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    loaded_lr = config.get('learning_rate', LEARNING_RATE)
                    print(f"  ✓ Learning rate loaded: {loaded_lr:.9f}")
                
                # Load optimizer state if available
                optimizer_path = filepath.replace('_full.keras', '_optimizer.keras')
                if os.path.exists(optimizer_path):
                    loaded_optimizer = Adam(learning_rate=loaded_lr)
                    loaded_optimizer.build(model_variables=loaded_model.trainable_variables)
                    loaded_optimizer.load_weights(optimizer_path)
                    print(f"  ✓ Optimizer state loaded")
                
                # Verify encoder works with a test pass
                test_input = tf.random.normal((1, 28, 28, 1), dtype=tf.float32)
                try:
                    _ = loaded_model(test_input, training=False)
                    print("  ✓ Full model loaded successfully")
                    return loaded_model, loaded_optimizer, loaded_lr, True
                except Exception as e:
                    print(f"  ✗ Test forward pass failed: {e}")
                    return None, None, LEARNING_RATE, False
            else:
                print(f"  ✗ Wrong model type: {type(loaded_model)}")
                return None, None, LEARNING_RATE, False
        except Exception as e:
            print(f"  ✗ Full .keras load failed: {e}")
    
    # Strategy 2: Try weights + config
    config_path = filepath.replace('_full.keras', '_config.json')
    weights_path = filepath.replace('_full.keras', '_weights.keras')
    optimizer_path = filepath.replace('_full.keras', '_optimizer.keras')
    
    if os.path.exists(config_path) and os.path.exists(weights_path):
        try:
            print("  Trying config + weights load...")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            loaded_lr = config.get('learning_rate', LEARNING_RATE)
            print(f"  ✓ Learning rate loaded: {loaded_lr:.9f}")
            
            # Rebuild model with encoder
            model = MultiHopHyperRetriever(
                enc=encoder_layer,
                num_hops=config['num_hops'],
                target_dim=config['target_dim'],
                hyper_arch=config['hyper_arch'],
                output_dim=config['output_dim'],
                initial_temperature=config['initial_temperature'],
                saved_learning_rate=loaded_lr
            )
            
            # Build variables with dummy pass
            dummy_input = tf.random.normal((1, 28, 28, 1), dtype=tf.float32)
            _ = model(dummy_input, training=False)
            
            # Load weights
            model.load_weights(weights_path)
            
            # Load optimizer state
            loaded_optimizer = Adam(learning_rate=loaded_lr)
            loaded_optimizer.build(model_variables=model.trainable_variables)
            if os.path.exists(optimizer_path):
                loaded_optimizer.load_weights(optimizer_path)
                print(f"  ✓ Optimizer state loaded")
            
            print("  ✓ Config + weights loaded successfully")
            return model, loaded_optimizer, loaded_lr, True
        except Exception as e:
            print(f"  ✗ Config + weights load failed: {e}")
    
    print("  ✗ All loading strategies failed!")
    return None, None, LEARNING_RATE, False


# ---------------------------------------------------------
# 4b. Build HQE Model & Load Weights if Exist
# ---------------------------------------------------------

MODEL_LOADED = False
frozen_enc_layer = FrozenEncoderLayer(loaded_encoder)

# *** FIXED: Initialize MEM_BANK_VECS placeholder with enough vectors for top_k ***
# Must have at least NUM_NEIGHBORS vectors to avoid TopKV2 error during dummy pass
MEM_BANK_VECS = tf.constant(np.zeros((NUM_NEIGHBORS, EMBEDDING_DIM), dtype=np.float32))
MEM_BANK_LABELS = tf.constant(np.zeros((NUM_NEIGHBORS, NUM_ACTIONS), dtype=np.float32))

# Always build fresh architecture first (for fallback)
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
print(f"  - Visual Centroids: {NUM_VISUAL_CENTROIDS} ({'LOADED' if CENTROIDS_EXIST else 'NEW'})")

# *** LOAD MODEL ***
hqe_model_for_encoding = None
HQE_MODEL_AVAILABLE = False
loaded_optimizer = None
loaded_lr = LEARNING_RATE

# Register custom objects before loading
keras.utils.get_custom_objects().update(CUSTOM_OBJECTS)

if LOAD_PREVIOUS_MODEL and os.path.exists(SAVE_PATH_HQE_FULL):
    print(f"\n[LOAD] Full model found at {SAVE_PATH_HQE_FULL}")
    
    # Use custom load function with multiple strategies
    loaded_model, loaded_optimizer, loaded_lr, load_success = load_hqe_model(
        SAVE_PATH_HQE_FULL,
        frozen_enc_layer,
        custom_objects=CUSTOM_OBJECTS
    )
    
    if load_success and loaded_model is not None:
        # Double-check encoder is set
        if loaded_model.check_encoder():
            retriever_branch = loaded_model
            MODEL_LOADED = True
            HQE_MODEL_AVAILABLE = True
            hqe_model_for_encoding = retriever_branch
            print("✓ Model loaded successfully!")
            print(f"✓ Learning rate from previous run: {loaded_lr:.9f}")
            
            # Run comprehensive verification
            verify_passed = verify_model_loading(retriever_branch, "HQE Retriever")
            
            if not verify_passed:
                print("⚠ Verification failed - model may have issues!")
                print("Continuing with fresh model initialization...")
                MODEL_LOADED = False
                HQE_MODEL_AVAILABLE = False
                loaded_optimizer = None
                retriever_branch = MultiHopHyperRetriever(
                    enc=frozen_enc_layer, 
                    num_hops=NUM_HOPS, 
                    target_dim=EMBEDDING_DIM, 
                    hyper_arch=TARGET_NET_ARCH,
                    output_dim=EMBEDDING_DIM,
                    initial_temperature=INIT_TEMP
                )
        else:
            print("✗ Encoder not properly set after load!")
            MODEL_LOADED = False
    else:
        print("✗ Model load failed - using fresh initialization")
        MODEL_LOADED = False
elif LOAD_PREVIOUS_MODEL and os.path.exists(SAVE_PATH_HQE_WEIGHTS):
    print(f"\n[LOAD] Weights found at {SAVE_PATH_HQE_WEIGHTS}")
    print("Loading weights into fresh architecture...")
    try:
        # Create variables with dummy pass first
        dummy_input = tf.random.normal((1, 28, 28, 1), dtype=tf.float32)
        _ = retriever_branch(dummy_input, training=False)
        
        # Load weights
        retriever_branch.load_weights(SAVE_PATH_HQE_WEIGHTS)
        MODEL_LOADED = True
        HQE_MODEL_AVAILABLE = True
        hqe_model_for_encoding = retriever_branch
        
        # Try to load LR from config
        config_path = SAVE_PATH_HQE_WEIGHTS.replace('_weights.keras', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            loaded_lr = config.get('learning_rate', LEARNING_RATE)
            print(f"✓ Learning rate loaded from config: {loaded_lr:.9f}")
        
        print("✓ Weights loaded successfully!")
        
        # Run verification
        verify_passed = verify_model_loading(retriever_branch, "HQE Retriever")
        
        if not verify_passed:
            print("⚠ Verification failed after weights load!")
            MODEL_LOADED = False
            HQE_MODEL_AVAILABLE = False
            
    except Exception as e:
        print(f"✗ Weights load failed: {e}")
        print("Starting with fresh weights...")
        MODEL_LOADED = False
else:
    print("\n>>> No previous model found. Using FRESH MODEL for training")
    MODEL_LOADED = False

if MODEL_LOADED:
    print("\n>>> Using PREVIOUS MODEL/WEIGHTS as starting point")
    print(f">>> Learning Rate: {loaded_lr:.9f} (from previous run)")
else:
    print("\n>>> Using FRESH MODEL for training")
    print(f">>> Learning Rate: {LEARNING_RATE:.9f} (from config)")
    loaded_lr = LEARNING_RATE

# ---------------------------------------------------------
# 5. Short Term Memory Bank (STM) - Initialization (Persistent)
# ---------------------------------------------------------
if USING_STM:
    print(f"\n>>> Initializing Short Term Memory DB at {STM_DB_PATH}...")
    
    # *** FIXED: Remove deprecated Settings parameter ***
    stm_client = chromadb.PersistentClient(path=STM_DB_PATH)
    
    # *** CHANGED: Don't delete collection if it exists ***
    if PERSIST_STM_ACROSS_RUNS:
        try:
            stm_collection = stm_client.get_collection(STM_COLLECTION_NAME)
            existing_stm_count = get_collection_count(stm_collection)
            print(f"Existing STM collection found with {existing_stm_count} vectors")
        except:
            stm_collection = stm_client.get_or_create_collection(STM_COLLECTION_NAME)
            existing_stm_count = 0
            print("Created new STM collection")
    else:
        try: 
            stm_client.delete_collection(STM_COLLECTION_NAME)
        except: 
            pass
        stm_collection = stm_client.get_or_create_collection(STM_COLLECTION_NAME)
        existing_stm_count = 0
    
    stm_vecs_list = []   
    stm_labels_list = [] 
else:
    stm_collection = None
    existing_stm_count = 0

if USING_STM and existing_stm_count > 0:
    stm_results = stm_collection.get(include=['embeddings', 'metadatas'])
    GLOBAL_STM_VECS = np.array(stm_results['embeddings']).astype('float32')
    GLOBAL_STM_LABELS = []
    for m in stm_results['metadatas']:
        GLOBAL_STM_LABELS.append(ast.literal_eval(m['one_hot_vector']))
    GLOBAL_STM_LABELS = np.array(GLOBAL_STM_LABELS).astype('float32')
    print(f"\n✓ STM loaded for training: {len(GLOBAL_STM_VECS)} vectors")

# ---------------------------------------------------------
# 6. LTM INITIALIZATION (Persistent with FIFO Eviction)
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("LTM Initialization (Persistent with FIFO Eviction)")
print("_______________________________________________________________________")

# Initialize Chroma client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# *** CHANGED: Don't delete collection if it exists ***
if PERSIST_LTM_ACROSS_RUNS:
    try:
        collection = client.get_collection(COLLECTION_NAME)
        existing_count = get_collection_count(collection)
        print(f"Existing LTM collection found with {existing_count} vectors")
        LTM_EXISTS = True
    except:
        collection = client.get_or_create_collection(COLLECTION_NAME)
        existing_count = 0
        print("Created new LTM collection")
        LTM_EXISTS = False
else:
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection = client.get_or_create_collection(COLLECTION_NAME)
    existing_count = 0
    LTM_EXISTS = False

# Load existing LTM vectors if they exist
if LTM_EXISTS and existing_count > 0:
    print("Loading existing LTM vectors...")
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
    print(f"Loaded {len(db_vecs_raw)} existing LTM vectors")
    
    # Check if we need to seed more (only if below threshold)
    LTM_SEED_THRESHOLD = LTM_MAX_CAPACITY + 1  #Patch to "Skipping LTM Seeding (sufficient vectors exist)"
    SHOULD_SEED = existing_count < LTM_SEED_THRESHOLD

    # =========================================================
    # === ADD DEBUG CODE HERE (AFTER loading, BEFORE seeding) ===
    # =========================================================
    print(f"\n=== DEBUG: Existing LTM Labels Check ===")
    print(f"db_labels_raw shape: {db_labels_raw.shape}")
    print(f"First 5 one-hot vectors: {db_labels_raw[:5]}")
    print(f"Current extraction [:,0]: {db_labels_raw[:, 0][:5].astype(int)}")
    print(f"Correct argmax: {np.argmax(db_labels_raw, axis=1)[:5]}")
    print(f"==========================================\n")
    # =========================================================

else:
    SHOULD_SEED = True
    MEM_BANK_VECS = tf.constant(np.zeros((NUM_NEIGHBORS, EMBEDDING_DIM), dtype=np.float32))
    MEM_BANK_LABELS = tf.constant(np.zeros((NUM_NEIGHBORS, NUM_ACTIONS), dtype=np.float32))

# *** NEW: Use HQE for LTM Encoding if Available ***
USE_HQE_FOR_LTM_ENCODING = True #Can always be true - LTM_USE_FROZEN_ENCODER_FOR_INSERTION can override


# ---------------------------------------------------------
# 6b. LTM SEEDING (Now HQE is Available!)
# ---------------------------------------------------------
if SHOULD_SEED:
    print("\n_______________________________________________________________________")
    print("LTM Seeding: Identical to STM Strategy 1 (Pre-Training)")
    print("_______________________________________________________________________")

    # *** Choose encoder for LTM seeding ***
    # *** NEW: Check LTM_USE_FROZEN_ENCODER_FOR_INSERTION flag ***
    if (USE_HQE_FOR_LTM_ENCODING and 
        HQE_MODEL_AVAILABLE and 
        hqe_model_for_encoding is not None and
        not LTM_USE_FROZEN_ENCODER_FOR_INSERTION):  # ← NEW CHECK
        print("Using HQE model for LTM encoding (Q-space)...")
        USE_HQE_ENCODING = True
    else:
        print("Using frozen encoder for LTM encoding (Z-space)...")
        USE_HQE_ENCODING = False

    # Step 1: Encode All Training Data
    print("Encoding Training Data for LTM Seed...")
    
    if USE_HQE_ENCODING:
        Z_pool = encode_images(hqe_model_for_encoding, X_train_val, batch_size=256)
        print("  Encoder: HQE (Q-space - Multi-Hop Transformed)")
    else:
        Z_pool = encode_images(loaded_encoder, X_train_val, batch_size=256)
        print("  Encoder: Frozen CNN (Z-space - Base Encoding)")
    
    Z_pool_norm = tf.nn.l2_normalize(Z_pool, axis=1).numpy()

    # Step 2: Split (Identical Ratio to STM - 20% Val, 80% Candidates)
    n_total = len(Z_pool_norm)
    n_val = int(n_total * LTM_OPTIMIZATION_SUBSET_RATIO)  # 20%
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

    current_ltm_vecs = []  # List of arrays
    current_ltm_labels = []  # List of arrays
    
    # Load existing vectors if persistent
    if LTM_EXISTS and existing_count > 0:
        current_ltm_vecs = [db_vecs_raw]
        current_ltm_labels = [np.argmax(db_labels_raw, axis=1).astype(int)]
        print(f"Starting with {existing_count} existing LTM vectors")
    
    best_acc = 0.0
    global_insert_count = 0

    # ---------------------------------------------------------
    # *** FIXED: k-NN Accuracy (Handles Both Int & One-Hot Labels) ***
    # ---------------------------------------------------------
    def knn_accuracy(query_zs, query_labels, memory_zs, memory_labels, k=5, 
                    hqe_model=None, raw_images=None, temperature=1.0, num_classes=10):
        """
        Matches MultiHopHyperRetriever Memory Bank Retrieval EXACTLY:
        tf.math.top_k → Temperature → tf.nn.softmax → Weighted Label Sum
        """
        if len(memory_zs) == 0:
            return 0.0
        
        # === Encode queries through HQE if provided (matches inference) ===
        if hqe_model is not None and raw_images is not None:
            preds = []
            batch_size = 256
            for i in range(0, len(raw_images), batch_size):
                x_b = raw_images[i:i+batch_size]
                out = hqe_model(x_b, training=False)
                preds.extend(np.argmax(out.numpy(), axis=1))
            return accuracy_score(query_labels, preds)
        
        # === Convert everything to TensorFlow tensors (matches model) ===
        query_zs_tf = tf.convert_to_tensor(query_zs, dtype=tf.float32)
        query_zs_tf = tf.nn.l2_normalize(query_zs_tf, axis=1)
        
        if isinstance(memory_zs, list):
            memory_zs_arr = np.vstack(memory_zs)
        else:
            memory_zs_arr = memory_zs
        memory_zs_tf = tf.convert_to_tensor(memory_zs_arr, dtype=tf.float32)
        memory_zs_tf = tf.nn.l2_normalize(memory_zs_tf, axis=1)
        
        if isinstance(memory_labels, list):
            memory_labels_arr = np.concatenate(memory_labels)
        else:
            memory_labels_arr = memory_labels
        
        # === FIX: Convert integer labels to one-hot if needed ===
        memory_labels_arr = np.array(memory_labels_arr)
        if len(memory_labels_arr.shape) == 1:
            # Integer labels → One-hot
            memory_labels_onehot = np.zeros((len(memory_labels_arr), num_classes), dtype=np.float32)
            for i, label in enumerate(memory_labels_arr):
                memory_labels_onehot[i, int(label)] = 1.0
            memory_labels_tf = tf.convert_to_tensor(memory_labels_onehot, dtype=tf.float32)
        else:
            # Already one-hot
            memory_labels_tf = tf.convert_to_tensor(memory_labels_arr, dtype=tf.float32)
        
        # === Compute Cosine Similarity Matrix (EXACT MATCH) ===
        sim_matrix = tf.matmul(query_zs_tf, memory_zs_tf, transpose_b=True)
        
        # === Top-K Selection (EXACT MATCH - tf.math.top_k) ===
        values_main, indices_main = tf.math.top_k(sim_matrix, k=k)
        
        # === Temperature Scaling (EXACT MATCH) ===
        scaled_values_main = values_main / temperature
        
        # === Softmax Attention (EXACT MATCH - tf.nn.softmax) ===
        attn_weights_main = tf.nn.softmax(scaled_values_main, axis=1)
        
        # === Gather Neighbor Labels (EXACT MATCH - tf.gather) ===
        # Now returns [n_queries, k, num_classes]
        neighbor_labels_main = tf.gather(memory_labels_tf, indices_main)
        
        # === Weighted Label Sum (EXACT MATCH - tf.reduce_sum) ===
        # attn_weights_main: [n_queries, k] → [n_queries, k, 1]
        # neighbor_labels_main: [n_queries, k, num_classes]
        pred_main = tf.reduce_sum(tf.expand_dims(attn_weights_main, -1) * neighbor_labels_main, axis=1)
        
        # === Final Prediction ===
        preds = tf.argmax(pred_main, axis=1).numpy()
        
        return accuracy_score(query_labels, preds)

    print(f"DEBUG: MEM_BANK_VECS shape: {MEM_BANK_VECS.shape}")
    print(f"DEBUG: current_ltm_vecs total: {sum(len(v) for v in current_ltm_vecs)}")
    print(f"DEBUG: LTM_USE_HQE_FOR_RETRIEVAL: {LTM_USE_HQE_FOR_RETRIEVAL}")
    print(f"DEBUG: HQE_MODEL_AVAILABLE: {HQE_MODEL_AVAILABLE}")

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
                
                # Hard Filter: sim < threshold
                eligible_mask = (max_sims <= LTM_SIMILARITY_THRESHOLD_CAND)
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
            
            # === Batch (Size ie 64) ===
            batch_size = min(LTM_INSERT_BATCH_SIZE, len(eligible_vecs))
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
    id_counter = 0

    for batch_data in all_batches:
        # *** REMOVED: No longer break at capacity - will evict instead ***
        
        if patience_counter >= LTM_PATIENCE:
            print(f"  >>> LTM Patience Reached. Stopping Optimization.")
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
            keep_mask = (max_sims < LTM_SIMILARITY_THRESHOLD_KEEP)
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

        # === Validate on ALL Z_val (mixed classes) ===
        # *** FIXED: Validation queries should match training/inference retrieval ***
        if LTM_USE_HQE_FOR_RETRIEVAL and HQE_MODEL_AVAILABLE and hqe_model_for_encoding is not None:
            #use hqe model
            acc = knn_accuracy(
                None,
                Y_val_int, 
                temp_ltm_vecs_arr, 
                temp_ltm_labels_arr, 
                k=NUM_NEIGHBORS,
                hqe_model=hqe_model_for_encoding,
                raw_images=X_train_val[val_indices],
                num_classes=NUM_ACTIONS
            )
            print(f"  Validation: HQE queries vs {'HQE' if USE_HQE_ENCODING else 'Frozen'} memory")
        else:
            acc = knn_accuracy(
                Z_val,
                Y_val_int, 
                temp_ltm_vecs_arr, 
                temp_ltm_labels_arr, 
                k=NUM_NEIGHBORS,
                num_classes=NUM_ACTIONS
            )
            print(f"  Validation: Frozen queries vs Frozen memory")
        
        if acc > best_acc:
            # ACCEPT
            best_acc = acc
            current_ltm_vecs.append(batch_vecs)
            current_ltm_labels.append(batch_labels_int)
            
            # *** FIXED: Make room BEFORE inserting (FIFO eviction) ***
            make_room_for_insert(collection, len(batch_vecs), LTM_MAX_CAPACITY, "LTM")
            
            # *** FIXED: Generate unique IDs with timestamp to avoid duplicates ***
            current_timestamp = datetime.now().timestamp()
            ids_to_insert = []
            metadatas_to_insert = []
            for idx in range(len(batch_vecs)):
                unique_id = generate_unique_id("ltm_seed", global_insert_count + idx, current_timestamp)
                ids_to_insert.append(unique_id)
                
                gt_vec = [0]*NUM_ACTIONS
                gt_vec[int(batch_labels_int[idx])] = 1
                metadatas_to_insert.append({
                    "true_label": int(batch_labels_int[idx]), 
                    "one_hot_vector": str(gt_vec),
                    "insert_timestamp": current_timestamp + idx  # Unique timestamp per vector
                })
            
            # *** Suppress ChromaDB warnings by using upsert instead of add ***
            try:
                collection.upsert(
                    embeddings=batch_vecs.tolist(),
                    ids=ids_to_insert,
                    metadatas=metadatas_to_insert
                )
            except Exception as e:
                # Fallback to add if upsert fails
                collection.add(
                    embeddings=batch_vecs.tolist(),
                    ids=ids_to_insert,
                    metadatas=metadatas_to_insert
                )
            # === FIX: Reload from ChromaDB to sync state ===
            results = collection.get(include=['embeddings', 'metadatas'])
            current_ltm_vecs = [np.array(results['embeddings']).astype('float32')]
            current_ltm_labels = []
            for m in results['metadatas']:
                current_ltm_labels.append(np.argmax(ast.literal_eval(m['one_hot_vector'])))
            current_ltm_labels = [np.array(current_ltm_labels)]

            # Update MEM_BANK_VECS for next validation iteration
            MEM_BANK_VECS = tf.constant(current_ltm_vecs[0])
            MEM_BANK_LABELS = tf.constant(
                tf.keras.utils.to_categorical(current_ltm_labels[0], NUM_ACTIONS)
            )
            
            global_insert_count += len(batch_vecs)
            id_counter += len(batch_vecs)
            patience_counter = 0
            print(f"  Batch {batch_idx+1} ACCEPTED (Label {batch_label}). New Acc: {acc:.4f} | Total LTM: {get_collection_count(collection)}/{LTM_MAX_CAPACITY}")
        else:
            # REJECT
            patience_counter += 1
            print(f"  Batch {batch_idx+1} REJECTED (Label {batch_label}). Acc: {acc:.4f} | Patience: {patience_counter}/{LTM_PATIENCE}")
        
        batch_idx += 1

    end_time = time.time()
    print(f"\n>>> LTM Seeding Finished in {end_time - start_time:.2f} seconds.")
    print(f">>> Total Vectors Inserted into LTM: {global_insert_count}")
    print(f">>> Final LTM Accuracy (on Val Subset): {best_acc:.4f}")
    print(f">>> LTM Encoding Strategy: {'HQE (Q-space)' if USE_HQE_ENCODING else 'Frozen (Z-space)'}")
    print(f">>> HQE Available for Retrieval: {HQE_MODEL_AVAILABLE}")
    print(f">>> MEM_BANK_VECS shape: {MEM_BANK_VECS.shape}")
    print(f">>> current_ltm_vecs total: {sum(len(v) for v in current_ltm_vecs)}")

    # === ADD VACUUM HERE ===
    vacuum_chroma_database(CHROMA_DB_PATH)

else:
    print(">>> Skipping LTM Seeding (sufficient vectors exist)")

# ---------------------------------------------------------
# 7. Load LTM for Training (From Script A)
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

print(f"LTM Loaded: {len(db_vecs_raw)} vectors (Max: {LTM_MAX_CAPACITY})")

# ---------------------------------------------------------
# 8. Compile
# ---------------------------------------------------------
system_model = GuidedSystem(retriever_branch, VALUE_ENC_PATH)

# *** USE LOADED OPTIMIZER IF AVAILABLE ***
if loaded_optimizer is not None and MODEL_LOADED:
    print(f"\nUsing loaded optimizer (LR = {loaded_lr:.9f})")
    system_model.compile(
        optimizer=loaded_optimizer,  # ← Use loaded optimizer
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy']
    )
else:
    print(f"\nUsing fresh optimizer (LR = {loaded_lr:.9f})")
    system_model.compile(
        optimizer=Adam(learning_rate=loaded_lr),  # ← Use loaded LR
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
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

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=1, min_lr=1e-9, verbose=1)

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
n_opt_samples2 = int(len(X_train_val) * STM_LTM_MIX_OPTIMIZATION_RATIO)

opt_indices = np.random.choice(len(X_te), n_opt_samples, replace=False)
opt_indices2 = np.random.choice(len(X_train_val), n_opt_samples2, replace=False)

X_opt = X_te[opt_indices]
X_opt2 = X_train_val[opt_indices2]

y_opt_int = y_te_int[opt_indices]
y_opt_int2 = y_train_val_int[opt_indices2]

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

    # STRATEGY 1: Wrong + Low Similarity - Negative Incorrect Example
    if HYBRID_USE_LOW_SIM:
        is_low_sim = (sim_wrong <= STM_SIMILARITY_THRESHOLD_CAND)
        low_sim_idx = np.where(is_low_sim)[0]
        
        if len(low_sim_idx) > 0:
            for r in low_sim_idx:
                wrong_pred_cls = y_pred_cls[wrong_idx][r]  # 1. Get the incorrect prediction index
                neg_label = np.zeros(NUM_ACTIONS, dtype=np.float32); neg_label[wrong_pred_cls] = -1.0 * NEGATIVE_AVOIDANCE_WEIGHT  # 2. Create negative label to rapidly avoid incorrect predictions
                strategy1_candidates.append({
                    'type': 'low_sim', 
                    'image': x_wrong[r], 
                    'label_hot': neg_label,  # 3. Use neg_label instead of y_hot_wrong[r]
                    'label_int': int(y_int_wrong[r]), 
                    'sim': float(sim_wrong[r])
                })

            n_errors_low_sim += len(low_sim_idx)

    # STRATEGY 2: Wrong + Found Correct LTM Prototype - Positive Correct Example
    if HYBRID_USE_LTM_PROTO:
        if (STM_STORE_Q_NOT_Z_STRAT2):
            z_query = system_model.retriever(x_wrong, training=False, stm_vecs=None, stm_labels=None, return_intermediate=True)
            z_query = z_query.numpy()
        else:
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
        
        if STM_STORE_Q_NOT_Z_STRAT1:
            _, _, final_q, _ = system_model.retriever(
                batch_imgs, 
                training=False, 
                stm_vecs=None, 
                stm_labels=None, 
                return_intermediate=True
            )
            batch_vecs = final_q.numpy()
            print(f"    Strategy 1: Q-Storage (Multi-Hop Transformed) - Shape: {batch_vecs.shape}")
        else:
            batch_vecs = frozen_enc_layer(batch_imgs, training=False).numpy()
            print(f"    Strategy 1: Z-Storage (Frozen Encoder) - Shape: {batch_vecs.shape}")
        
        strat1_vecs.append(batch_vecs)
    strat1_vecs = np.vstack(strat1_vecs)
    
    for j in range(len(strat1_vecs)):
        vec = np.array(strat1_vecs[j], dtype=np.float32)
        candidate_vectors.append(vec)
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
        vec = gathered[j]
        if vec.ndim > 1:
            vec = vec.flatten()
        if len(vec) != EMBEDDING_DIM:
            print(f"    WARNING: Vector {j} has shape {vec.shape}, expected ({EMBEDDING_DIM},)")
            vec = vec[:EMBEDDING_DIM] if len(vec) >= EMBEDDING_DIM else np.pad(vec, (0, EMBEDDING_DIM - len(vec)))
        vec = np.array(vec, dtype=np.float32)
        
        rec = ltm_hits[int(g)]
        candidate_vectors.append(vec)
        candidate_labels_hot.append(g_labels[j])
        candidate_labels_int.append(int(np.argmax(g_labels[j])))
        candidate_counts.append(rec['count'])
        candidate_sims.append(float(rec['best_sim']))
        candidate_sources.append('ltm_proto')
    
    print(f">>> Strategy 2 LTM prototypes: {len(hit_indices)}")

if STM_DEDUP_CANDIDATES and len(candidate_vectors) > 0:
    print("\nApplying cross-strategy deduplication...")
    
    shapes = [v.shape for v in candidate_vectors]
    unique_shapes = set(shapes)
    if len(unique_shapes) > 1:
        print(f"    WARNING: Inconsistent vector shapes detected: {unique_shapes}")
        for i, (vec, shape) in enumerate(zip(candidate_vectors, shapes)):
            if shape != (EMBEDDING_DIM,):
                print(f"    Fixing vector {i}: {shape} -> ({EMBEDDING_DIM},)")
                candidate_vectors[i] = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    cand_vecs_arr = np.zeros((len(candidate_vectors), EMBEDDING_DIM), dtype=np.float32)
    for i, vec in enumerate(candidate_vectors):
        cand_vecs_arr[i] = vec
    
    print(f"    Candidate array shape: {cand_vecs_arr.shape}")
    
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
# PASS 2: ITERATIVE STM OPTIMIZATION (FIXED - Continuous FIFO)
# =========================================================
print("\n_______________________________________________________________________")
print("PASS 2: Iterative STM Optimization (With Bootstrapping + Error Validation)")
print("_______________________________________________________________________")

if USING_STM and len(candidate_vectors) > 0:
    cand_vecs = np.array(candidate_vectors)
    cand_labels_hot = np.array(candidate_labels_hot)
    cand_labels_int = np.array(candidate_labels_int)
    cand_sims = np.array(candidate_sims)
    
    unique_labels = np.unique(cand_labels_int)
    
    # === COLLECT ALL BATCHES FIRST (Grouped by Label) ===
    print(f"Organizing {len(cand_vecs)} candidates into batches by label...")
    all_stm_batches = []
    
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
        l_sims = l_sims[l_sort_idx]
        
        num_label_batches = int(np.ceil(len(l_vecs) / STM_INSERT_BATCH_SIZE))
        for i in range(num_label_batches):
            start = i * STM_INSERT_BATCH_SIZE
            end = min(start + STM_INSERT_BATCH_SIZE, len(l_vecs))
            
            all_stm_batches.append({
                'vecs': l_vecs[start:end],
                'hots': l_hots[start:end],
                'ints': l_ints[start:end],
                'sims': l_sims[start:end],
                'label': int(label)
            })
    
    print(f">>> Total STM Batches Collected: {len(all_stm_batches)}")
    np.random.shuffle(all_stm_batches)
    print(">>> STM Batches Shuffled (Cross-Label)")
    
    # === PREPARE ERROR SUBSET FOR VALIDATION ===
    print("Identifying model errors on optimization subset...")
    opt_preds = []
    for i in range(0, len(X_opt), 256):
        x_b = X_opt[i:i+256]
        out = system_model.retriever(x_b, training=False, stm_vecs=None, stm_labels=None)
        opt_preds.extend(np.argmax(out.numpy(), axis=1))
    
    error_mask = (np.array(opt_preds) != y_opt_int)
    X_opt_errors = X_opt[error_mask]
    y_opt_errors = y_opt_int[error_mask]
    
    print(f">>> Found {len(X_opt_errors)} errors out of {len(X_opt)} samples")
    
    if len(X_opt_errors) < 10:
        print(">>> Too few errors to validate on. Using full X_opt instead.")
        X_opt_errors = X_opt
        y_opt_errors = y_opt_int
    
    # === PROCESS SHUFFLED BATCHES ===
    current_stm_vecs = []
    current_stm_labels = []
    
    # Load existing STM if persistent
    if PERSIST_STM_ACROSS_RUNS and existing_stm_count > 0:
        print("Loading existing STM vectors...")
        stm_results = stm_collection.get(include=['embeddings', 'metadatas'])
        current_stm_vecs = [np.array(stm_results['embeddings']).astype('float32')]
        current_stm_labels = []
        for m in stm_results['metadatas']:
            try:
                current_stm_labels.append(ast.literal_eval(m['one_hot_vector']))
            except:
                current_stm_labels.append([0]*10)

        current_stm_labels = [np.array(current_stm_labels).astype('float32')]
        print(f"Starting with {existing_stm_count} existing STM vectors")

    temp_acc_train_1 = calculate_accuracy_with_stm(system_model, X_opt2, y_opt_int2, [], [])
    print(f">>> Benchmarking STM Train Accuracy: {temp_acc_train_1:.4f}")

    baseline_acc = calculate_accuracy_with_stm(system_model, X_opt_errors, y_opt_errors, [], [])
    print(f">>> Baseline Accuracy (No STM, on errors only): {baseline_acc:.4f}")

    best_acc = baseline_acc
    no_improve_count = 0
    total_inserted = 0
    stm_id_counter = 0
    
    start_time = time.time()
    batch_idx = 0
    BOOTSTRAP_BATCHES = 10
    
    for batch_data in all_stm_batches:
        # *** REMOVED: No longer break at capacity - will evict instead ***
        
        if no_improve_count >= STM_PATIENCE:
            print(f"  >>> STM Patience Reached. Stopping Optimization.")
            break
        
        batch_vecs = batch_data['vecs']
        batch_labels_hot = batch_data['hots']
        batch_labels_int = batch_data['ints']
        batch_label = batch_data['label']
        
        # === SURPRISE GATE ===
        if len(current_stm_vecs) > 0 and total_inserted >= STM_BOOTSTRAP_TOTAL:
            stm_arr = np.vstack(current_stm_vecs)
            
            batch_vecs_norm = batch_vecs / np.linalg.norm(batch_vecs, axis=1, keepdims=True)
            stm_arr_norm = stm_arr / np.linalg.norm(stm_arr, axis=1, keepdims=True)
            
            sims = np.dot(batch_vecs_norm, stm_arr_norm.T)
            max_sims = np.max(sims, axis=1)
            
            keep_mask = (max_sims < STM_SIMILARITY_THRESHOLD_KEEP)
            if np.sum(keep_mask) == 0:
                print(f"  Batch {batch_idx+1} (Class {batch_label}): SKIPPED (All Redundant)")
                print(f"    Debug: Max sims range [{np.min(max_sims):.4f}, {np.max(max_sims):.4f}]")
                batch_idx += 1
                continue
            
            batch_vecs = batch_vecs[keep_mask]
            batch_labels_hot = batch_labels_hot[keep_mask]
            batch_labels_int = batch_labels_int[keep_mask]
            
            if len(batch_vecs) == 0:
                print(f"  Batch {batch_idx+1} (Class {batch_label}): SKIPPED (Empty After Filter)")
                batch_idx += 1
                continue
        
        # === Test & Accept ===
        temp_stm_vecs = current_stm_vecs + [batch_vecs] if current_stm_vecs else [batch_vecs]
        temp_stm_vecs_np = np.vstack(temp_stm_vecs)
        temp_stm_labels_np = np.vstack(current_stm_labels + [batch_labels_hot]) if current_stm_labels else batch_labels_hot
        
        temp_acc_test = calculate_accuracy_with_stm(system_model, X_opt_errors, y_opt_errors, temp_stm_vecs_np, temp_stm_labels_np)
        temp_acc_train = calculate_accuracy_with_stm(system_model, X_opt2, y_opt_int2, temp_stm_vecs_np, temp_stm_labels_np)
        
        test_rate_improved = temp_acc_test >= best_acc - 0.0001
        train_rate_improved = temp_acc_train >= temp_acc_train_1 - 0.0001
        
        print(f"  Test: {temp_acc_test:.4f} Train: {temp_acc_train:.4f}")

        if total_inserted < STM_BOOTSTRAP_TOTAL or np.logical_and(test_rate_improved, train_rate_improved).all():
            best_acc = max(temp_acc_test, best_acc)
            current_stm_vecs.append(batch_vecs)
            current_stm_labels.append(batch_labels_hot)
            total_inserted += len(batch_vecs)
            no_improve_count = 0
            accept_type = "BOOTSTRAP" if total_inserted < STM_BOOTSTRAP_TOTAL else "ACCEPTED"
            print(f"  Batch {batch_idx+1} (Class {batch_label}): {accept_type}. New Acc: {best_acc:.4f}")
            
            # *** FIXED: Make room BEFORE inserting (FIFO eviction) ***
            make_room_for_insert(stm_collection, len(batch_vecs), STM_MAX_CAPACITY, "STM")
            
            # *** FIXED: Generate unique IDs with timestamp to avoid duplicates ***
            current_timestamp = datetime.now().timestamp()
            ids_to_insert = []
            metadatas_to_insert = []
            for idx in range(len(batch_vecs)):
                unique_id = generate_unique_id("stm_opt", stm_id_counter + idx, current_timestamp)
                ids_to_insert.append(unique_id)
                
                gt_vec = [0]*10
                gt_vec[int(batch_labels_int[idx])] = 1
                metadatas_to_insert.append({
                    "true_label": int(batch_labels_int[idx]), 
                    "one_hot_vector": str(gt_vec),
                    "insert_timestamp": current_timestamp + idx
                })
            
            # *** Use upsert to avoid duplicate ID warnings ***
            try:
                stm_collection.upsert(
                    embeddings=batch_vecs.tolist(),
                    ids=ids_to_insert,
                    metadatas=metadatas_to_insert
                )
            except Exception as e:
                stm_collection.add(
                    embeddings=batch_vecs.tolist(),
                    ids=ids_to_insert,
                    metadatas=metadatas_to_insert
                )

             # === SYNC IN-MEMORY STATE WITH CHROMADB ===
            stm_results = stm_collection.get(include=['embeddings', 'metadatas'])
            current_stm_vecs = [np.array(stm_results['embeddings']).astype('float32')]
            current_stm_labels = []
            for m in stm_results['metadatas']:
                current_stm_labels.append(ast.literal_eval(m['one_hot_vector']))
            current_stm_labels = [np.array(current_stm_labels).astype('float32')]
            # =========================================
            
            stm_id_counter += len(batch_vecs)
        else:
            no_improve_count += 1
            print(f"  Batch {batch_idx+1} (Class {batch_label}): REJECTED. Patience: {no_improve_count}/{STM_PATIENCE}")
        
        batch_idx += 1
    
    end_time = time.time()
    print(f"\n>>> Optimization Finished in {end_time - start_time:.2f} seconds.")
    print(f">>> Total Samples Inserted into STM: {total_inserted}")
    print(f">>> Final Optimized STM Accuracy (on Error Subset): {best_acc:.4f}")

    # NEW (Load from ChromaDB - CORRECT):
    stm_results = stm_collection.get(include=['embeddings', 'metadatas'])
    stm_vecs_final = np.array(stm_results['embeddings']).astype('float32')
    stm_labels_final = []
    for m in stm_results['metadatas']:
        stm_labels_final.append(ast.literal_eval(m['one_hot_vector']))
    stm_labels_final = np.array(stm_labels_final).astype('float32')

    # *** VACUUM STM DATABASE AFTER OPTIMIZATION ***
    vacuum_chroma_database(STM_DB_PATH)
    
else:
    stm_vecs_final = np.empty((0, EMBEDDING_DIM))
    stm_labels_final = np.empty((0, 10))
    print(">>> No STM Optimization Performed.")

# === STM FINAL STATE DEBUG ===
print(f"\n========================================")
print(f"STM FINAL STATE CHECK")
print(f"========================================")

# Load directly from ChromaDB (not in-memory lists)
stm_results = stm_collection.get(include=['embeddings', 'metadatas'])
stm_vecs_db = np.array(stm_results['embeddings']).astype('float32')
stm_labels_db = []
for m in stm_results['metadatas']:
    stm_labels_db.append(ast.literal_eval(m['one_hot_vector']))
stm_labels_db = np.array(stm_labels_db).astype('float32')

print(f"STM vectors in ChromaDB: {len(stm_vecs_db)}")
print(f"STM capacity: {STM_MAX_CAPACITY}")
print(f"Utilization: {len(stm_vecs_db)/STM_MAX_CAPACITY*100:.1f}%")

stm_results_not_zero = (len(stm_vecs_db)/STM_MAX_CAPACITY*100) > 1

# Class distribution
if stm_results_not_zero:
    unique, counts = np.unique(np.argmax(stm_labels_db, axis=1), return_counts=True)
    print(f"\nClass Distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} ({count/len(stm_labels_db)*100:.1f}%)")

# Test accuracy with DB state (not in-memory)
print(f"\nAccuracy Test:")
test_acc = calculate_accuracy_with_stm(
    system_model, X_opt_errors, y_opt_errors,
    stm_vecs_db, stm_labels_db
)
print(f"STM Accuracy (from ChromaDB): {test_acc:.4f}")

# Compare in-memory vs ChromaDB
print(f"\nIn-Memory vs ChromaDB:")
print(f"  In-memory vectors: {sum(len(v) for v in current_stm_vecs)}")
print(f"  ChromaDB vectors: {len(stm_vecs_db)}")
if sum(len(v) for v in current_stm_vecs) != len(stm_vecs_db):
    print(f"  ⚠️ MISMATCH DETECTED!")
print(f"========================================\n")

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
# RECOVERY ANALYSIS
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
# 11. SAVE MODEL (Unified .keras + Config Fallback)
# ---------------------------------------------------------
print("\nSaving Trained Multi-Hop Hyper Retriever...")

# *** USE CUSTOM SAVE FUNCTION WITH OPTIMIZER ***
try:
    config_path, weights_path, optimizer_path = save_hqe_model(
        retriever_branch, 
        system_model.optimizer,  # ← Pass optimizer
        SAVE_PATH_HQE_FULL,
        save_weights_backup=True
    )
    print(f"\n✓ Model saved successfully!")
    print(f"  - Full model: {SAVE_PATH_HQE_FULL}")
    print(f"  - Config: {config_path}")
    print(f"  - Weights: {weights_path}")
    print(f"  - Optimizer: {optimizer_path}")
    print(f"  - Learning Rate: {system_model.optimizer.learning_rate.numpy():.9f}")
    
except Exception as e:
    print(f"✗ Save failed: {e}")
    retriever_branch.save_weights(SAVE_PATH_HQE_WEIGHTS)
    print(f"✓ Saved weights only to: {SAVE_PATH_HQE_WEIGHTS}")

print(f"Visual Centroids saved to: {SAVE_PATH_CENTROIDS}")

if ENABLE_CONSOLIDATION_EDA and eda_queries:
    print("\nSaving EDA Manifold Snapshots...")
    os.makedirs(EDA_SAVE_PATH, exist_ok=True)
    with open(os.path.join(EDA_SAVE_PATH, "query_evolution.pkl"), "wb") as f:
        pickle.dump(eda_queries, f)
    print(f"Saved EDA data to: {EDA_SAVE_PATH}")

# ---------------------------------------------------------
# 12. SUMMARY OF PERSISTENCE STATE
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("PERSISTENCE SUMMARY")
print("_______________________________________________________________________")

final_ltm_count = get_collection_count(collection)
final_stm_count = get_collection_count(stm_collection) if USING_STM else 0

print(f"LTM Collection: {final_ltm_count}/{LTM_MAX_CAPACITY} vectors")
print(f"STM Collection: {final_stm_count}/{STM_MAX_CAPACITY} vectors")
print(f"Model Weights Saved: {SAVE_PATH_HQE_WEIGHTS}")
print(f"Model Full Saved: {SAVE_PATH_HQE_FULL}")
print(f"Model Config Saved: {SAVE_PATH_HQE_CONFIG}")
print(f"Optimizer State Saved: {SAVE_PATH_HQE_FULL.replace('_full.keras', '_optimizer.keras')}")
print(f"Visual Centroids Saved: {SAVE_PATH_CENTROIDS}")
print(f"\nNext run will:")
print(f"  - Load existing LTM ({final_ltm_count} vectors)")
print(f"  - Load existing STM ({final_stm_count} vectors)")
print(f"  - Load previous model from .keras + config fallback")
print(f"  - Load previous optimizer state (with learning rate)")
print(f"  - Load existing visual centroids ({NUM_VISUAL_CENTROIDS} centroids)")
print(f"  - Use HQE model for LTM encoding (if weights exist)")
print(f"  - Append new vectors with FIFO eviction (continuous learning)")
print(f"\n*** ABLATION STUDY FLAGS ***")
print(f"  - LTM_USE_FROZEN_ENCODER_FOR_INSERTION: {LTM_USE_FROZEN_ENCODER_FOR_INSERTION}")
print(f"  - LTM_USE_HQE_FOR_RETRIEVAL: {LTM_USE_HQE_FOR_RETRIEVAL}")
print(f"\n*** LEARNING RATE ***")
print(f"  - Current LR: {system_model.optimizer.learning_rate.numpy():.9f}")
print(f"  - Will persist to next run: YES")

print("\n_______________________________________________________________________")
print("Training Complete!")
print("_______________________________________________________________________")