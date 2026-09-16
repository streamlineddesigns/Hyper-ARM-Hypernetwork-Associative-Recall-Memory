#---------------------------------------------------------
#   Copyright 2026 @Streamlined Designs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
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
# Hyperspherical Prototypes
# Prototype Mapping LUT (Dataset -> Class -> Prototype)
# DPAD: QE + VE + DE Branches (3-Branch Ensemble)
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
import itertools

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
STM_SIMILARITY_THRESHOLD_CAND = 0.85
STM_SIMILARITY_THRESHOLD_KEEP = 1.0

# Model Paths (Unified Weights Only)
ENCODER_PATH = "./saved_cnne_model_dir"
# *** VALUE ENC REMOVED FOR DPAD COMPATIBILITY ***
# *** UPDATED: Single .keras weights file + Full model ***
SAVE_PATH_HQE_WEIGHTS = "./saved_hqe_hyper_multi_hop_weights.keras"
SAVE_PATH_HQE_FULL = "./saved_hqe_hyper_multi_hop_full.keras"
SAVE_PATH_HQE_CONFIG = "./saved_hqe_hyper_multi_hop_config.json"

# *** NEW: Visual Centroids Path ***
SAVE_PATH_CENTROIDS = "./saved_visual_centroids.npy"

# *** NEW: Hyperspherical Prototype Config ***
PROTOTYPE_COUNT = 100               # Total pool of prototypes on hypersphere
PROTOTYPE_DIM = 128                 # Can differ from EMBEDDING_DIM (e.g., 256)
PROTOTYPE_SAVE_PATH = "./saved_hyperspherical_prototypes.npy"
PROTOTYPE_LUT_PATH = "./prototype_mapping_lut.json"
PROTOTYPE_OPTIMIZATION_EPOCHS = 500 # For repulsion optimization
PROTOTYPE_OPTIMIZATION_LR = 0.01
CURRENT_DATASET_NAME = "fashion_mnist" # Change per dataset run

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
STM_PATIENCE = 8
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
# *** NUM_CLASSES REMOVED - Calculated Dynamically from Data ***
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
LTM_CONFIDENCE_THRESHOLD = 1.0     # Below this = increase STM weight
STM_MAX_WEIGHT = 1.0
STM_MIN_WEIGHT = 0.5
WEIGHT_BOOST_FACTOR = 1.0 #1.0=Linear, 1.25=EaseInOutSine, 1.5=EaseInOutQuad, 2.0=EasInOutExpo (Linear approximations)
LOG_CONFIDENCE_SCORES = False

# *** DPAD BRANCH FLAGS ***
USE_VE_BRANCH = True   # Value Encoder Branch (Legacy naming)
USE_DE_BRANCH = True   # Directional Encoder Branch (DPAD alignment)
USE_CE_BRANCH = True   # C Encoder Branch (Attention over submodules)
CE_OUTPUT_DIM = 3

# Add this after your CONFIGURATION section:
GLOBAL_STM_VECS = None
GLOBAL_STM_PROTOS = None

#Grid Search on the selected hyperparameters
HYPERPARM_GRID_SEARCH = False
GRID_SEARCH_INDEX = 0
GRID_SEARCH_INDEX_FILE_PATH = "./grid_search_index_file.txt"
GRID_SEARCH_BEST_INDEX_FILE_PATH  = "./grid_search_best_index_file.txt"
GRID_SEARCH_BEST_ACCURACY_FILE_PATH = "./grid_search_best_accuracy_file.txt"

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
# *** NEW: Hyperspherical Prototype Generation (FIXED) ***
# ---------------------------------------------------------
def generate_hyperspherical_prototypes(count, dim, optimization_epochs=500, lr=0.01):
    """
    Generates evenly distributed prototypes on a hypersphere using repulsion optimization.
    """
    print(f"Generating {count} hyperspherical prototypes (Dim={dim})...")
    
    # Initialize random vectors on hypersphere
    prototypes = tf.Variable(tf.random.normal((count, dim)), dtype=tf.float32, trainable=True)
    prototypes.assign(tf.nn.l2_normalize(prototypes, axis=1))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # Repulsion Loss: Minimize pairwise cosine similarity
    def repulsion_step():
        with tf.GradientTape() as tape:
            tape.watch(prototypes)  # <--- EXPLICITLY WATCH THE VARIABLE
            # Normalize
            norm_prots = tf.nn.l2_normalize(prototypes, axis=1)
            # Cosine Similarity Matrix
            sim_matrix = tf.matmul(norm_prots, norm_prots, transpose_b=True)
            # Mask diagonal (self-similarity)
            mask = 1.0 - tf.eye(count)
            # Loss: Sum of squared similarities (push apart)
            loss = tf.reduce_sum((sim_matrix * mask) ** 2)
        
        gradients = tape.gradient(loss, [prototypes])
        optimizer.apply_gradients(zip(gradients, [prototypes]))
        
        # Re-normalize after step to ensure hypersphere constraint
        prototypes.assign(tf.nn.l2_normalize(prototypes, axis=1))
        return loss

    for i in range(optimization_epochs):
        loss = repulsion_step()
        if (i + 1) % 100 == 0:
            print(f"  Prototype Opt Epoch {i+1}/{optimization_epochs}, Loss: {loss.numpy():.4f}")
    
    return prototypes.numpy()

# ---------------------------------------------------------
# *** NEW: Prototype Mapping Manager (Persistent LUT) ***
# ---------------------------------------------------------
class PrototypeMappingManager:
    def __init__(self, prototype_vectors, lut_path):
        self.prototype_vectors = prototype_vectors # Shape: (PROTOTYPE_COUNT, PROTOTYPE_DIM)
        self.lut_path = lut_path
        self.slot_lut = [False] * len(prototype_vectors) # False = Available
        self.dataset_lut = {} # { dataset_name: { class_idx: prototype_idx } }
        self.load()
    
    def load(self):
        if os.path.exists(self.lut_path):
            with open(self.lut_path, 'r') as f:
                data = json.load(f)
            self.slot_lut = data.get('slot_lut', [False] * len(self.prototype_vectors))
            self.dataset_lut = data.get('dataset_lut', {})
            print(f"Loaded Prototype LUT from {self.lut_path}")
        else:
            print("Initialized new Prototype LUT")
    
    def save(self):
        data = {
            'slot_lut': self.slot_lut,
            'dataset_lut': self.dataset_lut
        }
        with open(self.lut_path, 'w') as f:
            json.dump(data, f)
        print(f"Saved Prototype LUT to {self.lut_path}")
    
    def get_prototypes_for_dataset(self, dataset_name, num_classes):
        """
        Returns: 
          1. List of prototype vectors for classes 0..num_classes-1
          2. Mapping dict { class_idx: prototype_idx }
        """
        if dataset_name in self.dataset_lut:
            # Existing dataset
            class_map = self.dataset_lut[dataset_name]
            # Ensure keys are integers for lookup later
            int_class_map = {int(k): v for k, v in class_map.items()}
            prototypes = [self.prototype_vectors[idx] for idx in int_class_map.values()]
            return prototypes, int_class_map
        else:
            # New dataset: Assign unused prototypes
            available_indices = [i for i, used in enumerate(self.slot_lut) if not used]
            if len(available_indices) < num_classes:
                raise ValueError(f"Not enough unused prototypes! Need {num_classes}, have {len(available_indices)}")
            
            new_class_map = {}
            for cls_idx in range(num_classes):
                proto_idx = available_indices[cls_idx]
                self.slot_lut[proto_idx] = True
                new_class_map[cls_idx] = proto_idx 
            
            self.dataset_lut[dataset_name] = {str(k): v for k, v in new_class_map.items()}
            self.save()
            
            prototypes = [self.prototype_vectors[new_class_map[i]] for i in range(num_classes)]
            return prototypes, new_class_map

    def get_prototype_vector(self, dataset_name, class_idx):
        """Get specific prototype vector for a class"""
        if dataset_name not in self.dataset_lut:
            raise ValueError(f"Dataset {dataset_name} not registered")
        proto_idx = self.dataset_lut[dataset_name].get(str(class_idx))
        if proto_idx is None:
            raise ValueError(f"Class {class_idx} not mapped for {dataset_name}")
        return self.prototype_vectors[proto_idx]

#---------------------------------------------------------
#0. GENERATE CARTESIAN PRODUCT (Combinations)
#---------------------------------------------------------
def get_hyperparameter_combination(all_params, param_indices, set_index=0, verbose=True):
    """
    Generates the Cartesian product of selected hyperparameters and returns 
    the specific combination at 'set_index'.
    """
    
    # 1. Filter parameters based on indices provided
    if isinstance(param_indices[0], int):
        # Using integer indices
        selected_keys = [list(all_params.keys())[i] for i in param_indices]
    else:
        # Using string keys directly
        selected_keys = param_indices
    
    param_options = {key: all_params[key] for key in selected_keys if key in all_params}
    
    # Validate all keys exist
    missing_keys = [key for key in selected_keys if key not in all_params]
    if missing_keys:
        raise KeyError(f"Parameter keys not found in all_params: {missing_keys}")
    
    # 2. Extract names and lists
    param_names = list(param_options.keys())
    param_lists = list(param_options.values())
    
    # 3. Generate Cartesian Product
    raw_combinations = itertools.product(*param_lists)
    sets = [list(combo) for combo in raw_combinations]
    
    total_combos = len(sets)
    
    if total_combos == 0:
        raise ValueError("No combinations generated. Check input lists.")
        
    if set_index >= total_combos:
        raise IndexError(f"set_index {set_index} is out of range. Max index is {total_combos - 1}")

    # 4. Print Table (Dynamic based on number of parameters)
    if verbose:
        header = f"{'Set Index':<10} | " + " | ".join([f"{name:<15}" for name in param_names])
        print(header)
        print("-" * len(header))

        for i, combo in enumerate(sets):
            row_values = " | ".join([f"{str(val):<15}" for val in combo])
            print(f"Set {i:<5}    | {row_values}")
            
        print("-" * len(header))
        print(f"Total Combinations Generated: {total_combos}")
        print(f"Selected Set Index: {set_index}\n")

    # 5. Select and Map values back to names
    selected_combo = sets[set_index]
    result_dict = {name: val for name, val in zip(param_names, selected_combo)}
    
    return result_dict


# =========================================================
# PARAMETER DEFINITIONS (All in one place)
# =========================================================

ALL_HYPERPARAMS = {
    'STM_SIM_CAND':      [0.85, 0.75, 0.65],      # Index 0
    'STM_SIM_KEEP':      [1.0, 0.9, 0.8],         # Index 1
    'STM_LTM_M_SIM':     [0.7, 0.6, 0.5],         # Index 2
    'NEG_AVOIDANCE':     [1.25, 1.0, 0.75],       # Index 3
    'LTM_THRESH':        [1.0, 0.75, 0.5],        # Index 4
    'STM_MAX_W':         [1.0, 0.75, 0.5],        # Index 5
    'STM_MIN_W':         [0.5, 0.25, 0.0],        # Index 6
    'BOOST_FACT':        [2.0, 1.5, 1.0]          # Index 7
}

# =========================================================
# SELECT WHICH PARAMETERS TO INCLUDE IN GRID SEARCH
# =========================================================
def save_variable(filename, variable):
    with open(filename, 'w') as f:
        json.dump(variable, f)

def load_variable(filename, default_val):
    if not os.path.exists(filename):
        return default_val
    with open(filename, 'r') as f:
        return json.load(f)

if HYPERPARM_GRID_SEARCH:
    #change config for grid search
    EPOCHS = 0
    STM_PATIENCE = 64

    # OPTION 1: Use integer indices (based on order in ALL_HYPERPARAMS)
    # params_to_include = [4, 5, 6, 7]  # LTM_THRESH, STM_MAX_W, STM_MIN_W, BOOST_FACT
    
    # OPTION 2: Use parameter names directly (more readable!)
    params_to_include = ['LTM_THRESH', 'STM_MAX_W', 'STM_MIN_W', 'BOOST_FACT']
    #params_to_include = ['STM_SIM_CAND', 'STM_SIM_KEEP', 'STM_LTM_M_SIM', 'NEG_AVOIDANCE']
    
    #increment the index & load/save at same time
    GRID_SEARCH_INDEX = load_variable(GRID_SEARCH_INDEX_FILE_PATH, 0)
    temp_index = GRID_SEARCH_INDEX + 1
    save_variable(GRID_SEARCH_INDEX_FILE_PATH, temp_index)
    
    # Get the specific combination
    selected_params = get_hyperparameter_combination(
        all_params=ALL_HYPERPARAMS,
        param_indices=params_to_include, 
        set_index=GRID_SEARCH_INDEX, 
        verbose=True
    )

    # Verify Output
    print("--- Final Assigned Variables ---")
    for k, v in selected_params.items():
        print(f"{k}: {v:.3f}")

# ---------------------------------------------------------
# 1. DATA PREPARATION (From Script A - 50/50 Split)
# ---------------------------------------------------------
print("_______________________________________________________________________")
print("Loading Data")
print("_______________________________________________________________________")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_full = np.concatenate((x_train, x_test), axis=0)
Y_full = np.concatenate((y_train, y_test), axis=0)

# *** DYNAMIC NUM_CLASSES (Removed hardcoded config) ***
NUM_CLASSES = len(np.unique(Y_full))

# Check PROTOTYPE_COUNT is sufficient
if PROTOTYPE_COUNT < NUM_CLASSES:
    raise ValueError(f"PROTOTYPE_COUNT ({PROTOTYPE_COUNT}) < NUM_CLASSES ({NUM_CLASSES}). Increase PROTOTYPE_COUNT in config.")

X_processed = X_full.reshape(X_full.shape[0], 28, 28, 1).astype('float32') / 255.0

indices = np.arange(len(X_processed))

# SINGLE SPLIT: 50% Train/Val, 50% Test (From Script A)
idx_train_val, idx_test, _, _ = train_test_split(indices, Y_full, test_size=0.5, stratify=Y_full)

X_train_val = X_processed[idx_train_val]
y_train_val_int = Y_full[idx_train_val]
source_ids_train_val = idx_train_val.copy()

X_te = X_processed[idx_test]
y_te_int = Y_full[idx_test]

# Shuffle training data
shuffle_idx = np.random.permutation(len(X_train_val))
X_train_val = X_train_val[shuffle_idx]
y_train_val_int = y_train_val_int[shuffle_idx]
source_ids_train_val = source_ids_train_val[shuffle_idx]

# ---------------------------------------------------------
# 1b. HYPERSPHERICAL PROTOTYPE INITIALIZATION
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("Hyperspherical Prototype Initialization")
print("_______________________________________________________________________")

if os.path.exists(PROTOTYPE_SAVE_PATH):
    PROTOTYPE_VECTORS = np.load(PROTOTYPE_SAVE_PATH)
    print(f"Loaded existing prototypes from {PROTOTYPE_SAVE_PATH}")
else:
    PROTOTYPE_VECTORS = generate_hyperspherical_prototypes(
        PROTOTYPE_COUNT, PROTOTYPE_DIM, 
        PROTOTYPE_OPTIMIZATION_EPOCHS, PROTOTYPE_OPTIMIZATION_LR
    )
    np.save(PROTOTYPE_SAVE_PATH, PROTOTYPE_VECTORS)
    print(f"Saved new prototypes to {PROTOTYPE_SAVE_PATH}")

proto_manager = PrototypeMappingManager(PROTOTYPE_VECTORS, PROTOTYPE_LUT_PATH)

# *** NEW: Map Labels to Prototypes ***
# Get assigned prototypes for this dataset
assigned_prototypes, class_map = proto_manager.get_prototypes_for_dataset(
    CURRENT_DATASET_NAME, NUM_CLASSES
)
assigned_prototypes = np.array(assigned_prototypes).astype('float32')

# Convert Integer Labels to Prototype Vectors
def labels_to_prototypes(labels_int, proto_lookup):
    """Converts integer labels to prototype vectors"""
    proto_vecs = np.zeros((len(labels_int), PROTOTYPE_DIM), dtype=np.float32)
    for i, label in enumerate(labels_int):
        proto_vecs[i] = proto_lookup[label]
    return proto_vecs

# Apply to Train/Val/Test
Y_train_val_proto = labels_to_prototypes(y_train_val_int, assigned_prototypes)
Y_te_proto = labels_to_prototypes(y_te_int, assigned_prototypes)

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

# *** FIXED: Use PROTOTYPE_DIM for output dimension ***
TOTAL_PARAMS_PER_HOP = get_target_params_count(EMBEDDING_DIM, TARGET_NET_ARCH, PROTOTYPE_DIM)


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
    DPAD: QE + VE + DE Branches (3-Branch Ensemble)
    """
    def __init__(self, enc, num_hops, target_dim, hyper_arch, output_dim, ce_output_dim = 3,
             num_neighbors=5, initial_temperature=1.0, saved_learning_rate=None, 
             use_ve_branches=True, use_de_branches=True, use_ce_branches=True):
        super().__init__()
        self.enc = enc
        self.num_hops = num_hops
        self.target_dim = target_dim
        self.hyper_arch = hyper_arch
        self.output_dim = output_dim # PROTOTYPE_DIM
        self.num_neighbors = num_neighbors
        self.initial_temperature = initial_temperature
        self.saved_learning_rate = saved_learning_rate
        self._encoder_set = enc is not None
        self.use_ve_branches = use_ve_branches
        self.use_de_branches = use_de_branches
        self.use_ce_branches = use_ce_branches
        self.ce_output_dim = ce_output_dim
        
        # --- QE Branch (Query/Retrieval) ---
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

        # --- VE Branch (Value Encoder - Legacy Naming) ---
        if self.use_ve_branches:
            self.ve_hop_hypernets = [CentroidHypernetwork(
                output_param_count=get_target_params_count(target_dim, hyper_arch, output_dim),
                hop_id=i
            ) for i in range(num_hops)]
            self.ve_hop_target_nets = [DynamicTargetNetwork(
                arch_list=hyper_arch,
                output_dim=output_dim,
                hop_id=i
            ) for i in range(num_hops)]
        else:
            self.ve_hop_hypernets = None
            self.ve_hop_target_nets = None

        # --- CE Branch (C Encoder) ---
        if self.use_ce_branches:
            self.ce_hop_hypernets = [CentroidHypernetwork(
                output_param_count=get_target_params_count(target_dim, hyper_arch, ce_output_dim),
                hop_id=i
            ) for i in range(num_hops)]
            self.ce_hop_target_nets = [DynamicTargetNetwork(
                arch_list=hyper_arch,
                output_dim=ce_output_dim,
                hop_id=i
            ) for i in range(num_hops)]
        else:
            self.ce_hop_hypernets = None
            self.ce_hop_target_nets = None
        
        # --- DE Branch (Directional Encoder - DPAD) ---
        if self.use_de_branches:
            # Support Kernel Dim = K * K
            support_kernel_dim = num_neighbors * num_neighbors 
            # Params for MLP: Input(K) -> Hidden -> Output(K)
            de_param_count = get_target_params_count(num_neighbors, hyper_arch, num_neighbors)
            
            self.de_hop_hypernets = [CentroidHypernetwork( # Or new SupportHypernetwork
                output_param_count=de_param_count,
                hop_id=i
            ) for i in range(num_hops)]
            
            self.de_hop_target_nets = [DynamicTargetNetwork(
                arch_list=hyper_arch,
                output_dim=num_neighbors, # Output is Attention Weights (K)
                hop_id=i
            ) for i in range(num_hops)]
        else:
            self.de_hop_hypernets = None
            self.de_hop_target_nets = None
        
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
            'num_neighbors': self.num_neighbors,  # <--- ADDED
            'initial_temperature': self.initial_temperature,
            'saved_learning_rate': self.saved_learning_rate,
            'use_ve_branches': self.use_ve_branches,
            'use_de_branches': self.use_de_branches,
            'use_ce_branches': self.use_ce_branches,
            'ce_output_dim': self.ce_output_dim
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
        instance.num_neighbors = config.get('num_neighbors', 5) # <--- ADDED
        instance.initial_temperature = config.get('initial_temperature', 1.0)
        instance.saved_learning_rate = config.get('saved_learning_rate', None)
        instance.enc = None  # Will be replaced after loading
        instance._encoder_set = False  # Track encoder status
        instance.use_ve_branches = config.get('use_ve_branches', True)
        instance.use_de_branches = config.get('use_de_branches', True)
        instance.use_ce_branches = config.get('use_ce_branches', True)
        instance.ce_output_dim = config.get('ce_output_dim', 3)
        
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

        # VE Branch
        if instance.use_ve_branches:
            instance.ve_hop_hypernets = [CentroidHypernetwork(
                output_param_count=get_target_params_count(instance.target_dim, instance.hyper_arch, instance.output_dim),
                hop_id=i
            ) for i in range(instance.num_hops)]
            instance.ve_hop_target_nets = [DynamicTargetNetwork(
                arch_list=instance.hyper_arch,
                output_dim=instance.output_dim,
                hop_id=i
            ) for i in range(instance.num_hops)]
        else:
            instance.ve_hop_hypernets = None
            instance.ve_hop_target_nets = None

        # CE Branch
        if instance.use_ce_branches:
            instance.ce_hop_hypernets = [CentroidHypernetwork(
                output_param_count=get_target_params_count(instance.target_dim, instance.hyper_arch, instance.ce_output_dim),
                hop_id=i
            ) for i in range(instance.num_hops)]
            instance.ce_hop_target_nets = [DynamicTargetNetwork(
                arch_list=instance.hyper_arch,
                output_dim=instance.ce_output_dim,
                hop_id=i
            ) for i in range(instance.num_hops)]
        else:
            instance.ce_hop_hypernets = None
            instance.ce_hop_target_nets = None
        
        # DE Branch (Updated Dim)
        if instance.use_de_branches:
            de_param_count = get_target_params_count(instance.num_neighbors, instance.hyper_arch, instance.num_neighbors)
            instance.de_hop_hypernets = [CentroidHypernetwork(
                output_param_count=de_param_count, 
                hop_id=i
            ) for i in range(instance.num_hops)]
            instance.de_hop_target_nets = [DynamicTargetNetwork(
                arch_list=instance.hyper_arch, 
                output_dim=instance.num_neighbors, 
                hop_id=i
            ) for i in range(instance.num_hops)]
        else:
            instance.de_hop_hypernets = None
            instance.de_hop_target_nets = None
        
        return instance
    # =========================================================
        
    def get_temperature(self):
        temp = tf.exp(self.log_temp)
        return tf.clip_by_value(temp, MIN_TEMP, MAX_TEMP)

    def call(self, inputs, training=None, stm_vecs=None, stm_protos=None, 
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
        
        # === TRACKING: Final neighbors/protos for prediction (from last hop) ===
        final_neighbor_protos = None
        final_max_sim = None
        current_attention = None  # DE attention state (accumulates across hops)
        
        # === STEP 2: Multi-Hop with 1:1 CNN + Hypernetwork Per Hop ===
        # === UNIFIED LOOP: QE + Retrieval + DE + VE ===
        for i in range(self.num_hops):
            # --- QE Branch: Refine Query ---
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
            
            # QE Branch Sparse Mixture of Latent Experts :)    
            gen_params = self.hop_hypernets[i](ctx_vec)
            
            # Apply Generated Weights
            refined_delta = self.hop_target_nets[i](q_after_cnn, gen_params)
            current_q = q_after_cnn + refined_delta
            current_q = tf.linalg.l2_normalize(current_q, axis=1)
            
            if return_intermediate:
                intermediate_queries.append(current_q)
            
            # --- RETRIEVAL: Per-Hop Neighbor Selection (QE drives this) ---
            # Note: MEM_BANK_VECS will be set during LTM Initialization
            # MEM_BANK_PROTOTYPES replaces MEM_BANK_LABELS
            main_vecs_norm = tf.nn.l2_normalize(MEM_BANK_VECS, axis=1)
            sim_matrix_main = tf.matmul(current_q, main_vecs_norm, transpose_b=True)
            values_main, indices_main = tf.math.top_k(sim_matrix_main, k=NUM_NEIGHBORS)  
            max_sim_main = tf.reduce_max(values_main, axis=1)  # max un-normalized sim
            
            # Gather Prototype Vectors instead of Labels
            neighbor_protos_main = tf.gather(MEM_BANK_PROTOTYPES, indices_main) 
            # Shape: (Batch, K, EMBEDDING_DIM)
            
            # Gather Neighbor Vectors for DE Branch
            neighbor_vecs_main = tf.gather(MEM_BANK_VECS, indices_main)
            
            # Track final hop's retrieval for prediction
            final_neighbor_protos = neighbor_protos_main
            final_max_sim = max_sim_main
            
            # --- DE Branch: Attention Refinement (Per-Hop) ---
            if self.use_de_branches:
                # 1. Prepare Kernels
                # Query Kernel: Raw similarities (Batch, K)
                query_kernel = values_main 
                
                # Support Kernel: Neighbor vs Prototype Sim (Batch, K, K)
                neighbor_vecs_norm = tf.nn.l2_normalize(neighbor_vecs_main, axis=-1)
                neighbor_protos_norm = tf.nn.l2_normalize(neighbor_protos_main, axis=-1)
                
                # Compute KxK Sim Matrix per batch item
                support_kernel = tf.matmul(neighbor_vecs_norm, neighbor_protos_norm, transpose_b=True)  # (Batch, K, K)
                
                # === FIX: Explicitly define K*K dimension ===
                K_squared = self.num_neighbors * self.num_neighbors
                support_kernel_flat = tf.reshape(support_kernel, [tf.shape(inputs)[0], K_squared])  # (Batch, K*K)

                # 2. Initialize Attention State (Hop 0 only)
                if i == 0:
                    current_attention = query_kernel  # (Batch, K) - Start with raw similarities
                
                # 3. DE Hops (Refine Attention Weights)
                # DE Branch Sparse Mixture of Latent Experts :) (DPAD)
                gen_params = self.de_hop_hypernets[i](support_kernel_flat)
                
                # Target Network takes Query Kernel (Current Attention)
                refined_delta = self.de_hop_target_nets[i](current_attention, gen_params)
                
                # Residual Add on Attention Weights
                current_attention = current_attention + refined_delta

                # L2 normalize inside loop (keeps values bounded)
                current_attention = tf.linalg.l2_normalize(current_attention, axis=1)
                
                if return_intermediate:
                    if len(hop_data) > i:
                        hop_data[i]['attention_entropy'] = float(tf.reduce_mean(-tf.reduce_sum(current_attention * tf.log(current_attention + 1e-9), axis=1)).numpy())
                    else:
                        hop_data.append({
                            'hop_id': i,
                            'centroid_indices': best_idx.numpy() if hasattr(best_idx, 'numpy') else best_idx,
                            'hyper_params_mean': float(np.mean(gen_params.numpy() if hasattr(gen_params, 'numpy') else gen_params)),
                            'hyper_params_std': float(np.std(gen_params.numpy() if hasattr(gen_params, 'numpy') else gen_params))
                        })
            
            # --- VE Branch Sparse Mixture of Latent Experts :) ---
            if self.use_ve_branches:
                if i == 0:
                    current_v = tf.zeros((tf.shape(inputs)[0], self.output_dim), dtype=tf.float32)
                gen_params = self.ve_hop_hypernets[i](ctx_vec)
                refined_delta = self.ve_hop_target_nets[i](current_q, gen_params)
                current_v = current_v + refined_delta
                current_v = tf.linalg.l2_normalize(current_v, axis=1)

            # --- CE Branch Sparse Mixture of Latent Experts :) ---
            if self.use_ce_branches:
                if i == 0:
                    current_c = tf.zeros((tf.shape(inputs)[0], self.ce_output_dim), dtype=tf.float32)
                gen_params = self.ce_hop_hypernets[i](ctx_vec)
                refined_delta = self.ce_hop_target_nets[i](current_q, gen_params)
                current_c = current_c + refined_delta
                current_c = tf.linalg.l2_normalize(current_c, axis=1)
        
        # === Final Query ===
        final_q = current_q
        final_q = tf.nn.l2_normalize(final_q, axis=1)
        
        # Handle VE output
        if self.use_ve_branches:
            ve_output = current_v
        else:
            ve_output = tf.zeros((tf.shape(inputs)[0], self.output_dim), dtype=tf.float32)

        # Handle CE output
        if self.use_ce_branches:
            ce_output = current_c
        else:
            ce_output = tf.zeros((tf.shape(inputs)[0], self.ce_output_dim), dtype=tf.float32)

        ce_output = tf.nn.softmax(ce_output, axis=-1)
        # Extract each weight explicitly (no loop)
        w_qe = ce_output[:, 0]  # QE weight
        w_ve = ce_output[:, 1]  # VE weight
        w_de = ce_output[:, 2]  # DE weight
        w_qe_exp = tf.expand_dims(w_qe, axis=1)
        w_ve_exp = tf.expand_dims(w_ve, axis=1)
        w_de_exp = tf.expand_dims(w_de, axis=1)
        
                
        # === ENCODE ONLY MODE: Skip retrieval entirely ===
        if encode_only:
            if return_intermediate:
                return final_q, intermediate_queries, hop_data
            return final_q
        
        # Add noise during training (From Script A)
        if training:
            noise = tf.random.normal(shape=tf.shape(final_q), mean=0.0, stddev=0.01)
            final_q = final_q + noise
        
        current_temp = self.get_temperature()
        scaled_values_main = values_main / current_temp 
        attn_weights_main = tf.nn.softmax(scaled_values_main, axis=-1)
        pred_main = tf.reduce_sum(tf.expand_dims(attn_weights_main, -1) * final_neighbor_protos, axis=1)

        # === STEP 4: Apply Final Attention (Softmax ONCE at end) ===
        # Convert L2-normalized attention to probabilities
        if self.use_de_branches:
            scaled_values_de = current_attention / current_temp
            current_attention = tf.nn.softmax(scaled_values_de, axis=-1)
            pred_de = tf.reduce_sum(tf.expand_dims(current_attention, -1) * final_neighbor_protos, axis=1)
            de_output = pred_de
            
        else:
            # Fallback if DE disabled (use original retrieval weights)
            de_output = tf.zeros((tf.shape(inputs)[0], self.output_dim), dtype=tf.float32)
        
        # Shape: (Batch, EMBEDDING_DIM)
        pred_final = pred_main

        #training only
        if GLOBAL_STM_VECS is not None:
            stm_vecs = GLOBAL_STM_VECS
            stm_protos = GLOBAL_STM_PROTOS

        # === STEP 5: STM Retrieval (From Script A) ===
        if stm_vecs is not None and tf.shape(stm_vecs)[0] > 0:
            stm_vecs_norm = tf.nn.l2_normalize(stm_vecs, axis=1)
            sim_matrix_stm = tf.matmul(final_q, stm_vecs_norm, transpose_b=True)
            k_stm = tf.minimum(NUM_NEIGHBORS, tf.shape(stm_vecs_norm)[0])
            values_stm, indices_stm = tf.math.top_k(sim_matrix_stm, k=k_stm)    
            max_sim_stm = tf.reduce_max(values_stm, axis=1)  # max un-normalized sim        
            current_temp = self.get_temperature()
            scaled_values_stm = values_stm / current_temp 
            attn_weights_stm = tf.nn.softmax(scaled_values_stm, axis=1)
            
            # Gather STM Prototypes
            neighbor_protos_stm = tf.gather(stm_protos, indices_stm)
            pred_stm = tf.reduce_sum(tf.expand_dims(attn_weights_stm, -1) * neighbor_protos_stm, axis=1)
            
            ltm_confidence = final_max_sim  # Shape: [batch_size]
            stm_confidence = max_sim_stm  # Shape: [batch_size]

            # === DYNAMIC WEIGHTING BASED ON LTM CONFIDENCE ===
            # *** FIXED: Removed [0] index and if gate. Now works per-sample. ***
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

            # Optional: Log weighting for debugging
            if LOG_CONFIDENCE_SCORES:
                tf.print(f"  [Conf] LTM={tf.reduce_mean(final_max_sim):.3f} STM={tf.reduce_mean(max_sim_stm):.3f}", summarize=-1) 

            # 5. Compute the final prediction safely
            stm_prediction = pred_stm * stm_weight
            ltm_prediction = pred_main * ltm_weight
            pred_final = (ltm_prediction + stm_prediction)

        else:
            pred_final = pred_main

        # === DPAD: 3-Branch Ensemble (QE + VE + DE) ===
        # Normalize each branch output
        pred_final = tf.nn.l2_normalize(pred_final, axis=1)
        ve_output = tf.nn.l2_normalize(ve_output, axis=1)
        de_output = tf.nn.l2_normalize(de_output, axis=1)
        
        # Equal weighting (0.333 each)
        branch_count = 1  # QE always present
        if self.use_ve_branches:
            branch_count += 1
        if self.use_de_branches:
            branch_count += 1
        
        qe_weight = w_qe_exp
        ve_weight = w_ve_exp if self.use_ve_branches else 0.0
        de_weight = w_de_exp if self.use_de_branches else 0.0
        total_weight = qe_weight + ve_weight + de_weight
        qe_weight /= total_weight
        ve_weight /= total_weight
        de_weight /= total_weight

        pred_final = pred_final * qe_weight
        if self.use_ve_branches:
            pred_final = pred_final + (ve_output * ve_weight)
        if self.use_de_branches:
            pred_final = pred_final + (de_output * de_weight)
        
        pred_final = tf.nn.l2_normalize(pred_final, axis=1)  # Ensure output is on hypersphere

        if return_intermediate:
            if return_sim:
                return {'predictions': pred_final, 'max_similarity': final_max_sim}, intermediate_queries, final_q, hop_data
            return pred_final, intermediate_queries, final_q, hop_data
        else:
            if return_sim:
                return {'predictions': pred_final, 'max_similarity': final_max_sim}
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
    def __init__(self, retriever):
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
        
        # *** VALUE ENCODER REMOVED FOR DPAD COMPATIBILITY ***
        
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
        'CentroidHypernetwork': NUM_HOPS * (1 + (1 if model.use_ve_branches else 0) + (1 if model.use_de_branches else 0)),
        'DynamicTargetNetwork': NUM_HOPS * (1 + (1 if model.use_ve_branches else 0) + (1 if model.use_de_branches else 0)),
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
        
        if output.shape == (2, PROTOTYPE_DIM): # Updated for Prototypes
            print(f"  ✓ Output shape correct")
        else:
            print(f"  ✗ Output shape incorrect! Expected (2, {PROTOTYPE_DIM})")
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

    # 9. Check VE Branch Configuration
    print("\n[9] VE Branch Configuration Check:")
    if hasattr(model, 'use_ve_branches'):
        print(f"  ✓ use_ve_branches: {model.use_ve_branches}")
    else:
        print(f"  ✗ use_ve_branches attribute missing!")
        verification_passed = False

    # 10. Check CE Branch Configuration
    print("\n[9] CE Branch Configuration Check:")
    if hasattr(model, 'use_ce_branches'):
        print(f"  ✓ use_ce_branches: {model.use_ce_branches}")
    else:
        print(f"  ✗ use_ce_branches attribute missing!")
        verification_passed = False

    # 11. Check DE Branch Configuration (DPAD)
    print("\n[10] DE Branch Configuration Check:")
    if hasattr(model, 'use_de_branches'):
        print(f"  ✓ use_de_branches: {model.use_de_branches}")
        if model.use_de_branches and hasattr(model, 'de_hop_target_nets'):
            # Check output dim is NUM_NEIGHBORS (Attention Size), not PROTOTYPE_DIM
            expected_dim = model.num_neighbors 
            actual_dim = model.de_hop_target_nets[0].output_dim
            if actual_dim == expected_dim:
                print(f"  ✓ DE Target Net Output Dim: {actual_dim} (Attention Size)")
            else:
                print(f"  ✗ DE Target Net Output Dim Mismatch! Expected {expected_dim}, got {actual_dim}")
                verification_passed = False
    else:
        print(f"  ✗ use_de_branches attribute missing!")
        verification_passed = False


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
        'num_neighbors': model.num_neighbors,
        'hyper_arch': model.hyper_arch,
        'output_dim': model.output_dim,
        'initial_temperature': float(model.initial_temperature),
        'learning_rate': float(optimizer.learning_rate.numpy()),
        'optimizer_type': type(optimizer).__name__,
        'use_ve_branches': model.use_ve_branches,
        'use_de_branches': model.use_de_branches,
        'use_ce_branches': model.use_ce_branches,
        'ce_output_dim': model.ce_output_dim,
    }
    
    config_path = filepath.replace('_full.keras', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Config saved to {config_path}")
    print(f"  ✓ Learning rate saved: {config['learning_rate']:.9f}")
    print(f"  ✓ use_ve_branches saved: {config['use_ve_branches']}")  # Debug
    print(f"  ✓ use_ce_branches saved: {config['use_ce_branches']}")  # Debug
    print(f"  ✓ use_de_branches saved: {config['use_de_branches']}")  # Debug

    
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


def load_hqe_model(filepath, encoder_layer, custom_objects=None, enable_ve=True, enable_de=True):
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
                    print(f"  ✓ use_ve_branches loaded: {config.get('use_ve_branches', True)}")
                    print(f"  ✓ use_de_branches loaded: {config.get('use_de_branches', True)}")
                
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

            use_ve_from_config = config.get('use_ve_branches', True)
            use_de_from_config = config.get('use_de_branches', True)
            use_ce_from_config = config.get('use_ce_branches', True)
            
            print(f"  ✓ use_ve_branches from config: {use_ve_from_config}")
            print(f"  ✓ use_de_branches from config: {use_de_from_config}")

            
            # Rebuild model with encoder
            model = MultiHopHyperRetriever(
                enc=encoder_layer,
                num_hops=config['num_hops'],
                target_dim=config['target_dim'],
                hyper_arch=config['hyper_arch'],
                output_dim=config['output_dim'],
                num_neighbors=config.get('num_neighbors', NUM_NEIGHBORS), # <--- ADDED
                initial_temperature=config['initial_temperature'],
                saved_learning_rate=loaded_lr,
                use_ve_branches=use_ve_from_config,
                use_de_branches=use_de_from_config,
                use_ce_branches=use_ce_from_config,
                ce_output_dim=config['ce_output_dim'],
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
# *** FIXED: MEM_BANK_VECS uses EMBEDDING_DIM, MEM_BANK_PROTOTYPES uses PROTOTYPE_DIM ***
MEM_BANK_VECS = tf.constant(np.zeros((NUM_NEIGHBORS, EMBEDDING_DIM), dtype=np.float32))
MEM_BANK_PROTOTYPES = tf.constant(np.zeros((NUM_NEIGHBORS, PROTOTYPE_DIM), dtype=np.float32)) # Changed from LABELS

# Always build fresh architecture first (for fallback)
retriever_branch = MultiHopHyperRetriever(
    enc=frozen_enc_layer, 
    num_hops=NUM_HOPS, 
    target_dim=EMBEDDING_DIM, 
    hyper_arch=TARGET_NET_ARCH,
    output_dim=PROTOTYPE_DIM,
    num_neighbors=NUM_NEIGHBORS,  # <--- ADDED
    initial_temperature=INIT_TEMP,
    use_ve_branches=USE_VE_BRANCH,
    use_de_branches=USE_DE_BRANCH,
    use_ce_branches=USE_CE_BRANCH,
    ce_output_dim=CE_OUTPUT_DIM,
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
        custom_objects=CUSTOM_OBJECTS,
        enable_ve=USE_VE_BRANCH,
        enable_de=USE_DE_BRANCH,
        enable_ce=USE_CE_BRANCH
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
                    output_dim=PROTOTYPE_DIM,
                    initial_temperature=INIT_TEMP,
                    use_ve_branches=USE_VE_BRANCH,
                    use_de_branches=USE_DE_BRANCH,
                    use_ce_branches=USE_CE_BRANCH,
                    ce_output_dim=CE_OUTPUT_DIM,
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
    stm_protos_list = [] # Changed from LABELS
else:
    stm_collection = None
    existing_stm_count = 0

if USING_STM and existing_stm_count > 0:
    stm_results = stm_collection.get(include=['embeddings', 'metadatas'])
    GLOBAL_STM_VECS = np.array(stm_results['embeddings']).astype('float32')
    GLOBAL_STM_PROTOS = []
    for m in stm_results['metadatas']:
        GLOBAL_STM_PROTOS.append(json.loads(m['prototype_vector']))  # parse json string
    GLOBAL_STM_PROTOS = np.array(GLOBAL_STM_PROTOS).astype('float32')
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
    
    db_protos_raw = []
    for m in results['metadatas']:
        try: 
            db_protos_raw.append(json.loads(m['prototype_vector']))  # Parse JSON string
        except: 
            db_protos_raw.append([0]*PROTOTYPE_DIM) 
            
    db_protos_raw = np.array(db_protos_raw).astype('float32')
    MEM_BANK_VECS = tf.constant(db_vecs_raw)
    MEM_BANK_PROTOTYPES = tf.constant(db_protos_raw)
    print(f"Loaded {len(db_vecs_raw)} existing LTM vectors")
    
    # Check if we need to seed more (only if below threshold)
    if HYPERPARM_GRID_SEARCH:
        LTM_SEED_THRESHOLD = True
    else:
        LTM_SEED_THRESHOLD = LTM_MAX_CAPACITY + 1  #Patch to "Skipping LTM Seeding (sufficient vectors exist)"
    SHOULD_SEED = existing_count < LTM_SEED_THRESHOLD

    # =========================================================
    # === ADD DEBUG CODE HERE (AFTER loading, BEFORE seeding) ===
    # =========================================================
    print(f"\n=== DEBUG: Existing LTM Prototypes Check ===")
    print(f"db_protos_raw shape: {db_protos_raw.shape}")
    print(f"First 5 prototype vectors (sum): {np.sum(db_protos_raw[:5], axis=1)}")
    print(f"==========================================\n")
    # =========================================================

else:
    SHOULD_SEED = True
    MEM_BANK_VECS = tf.constant(np.zeros((NUM_NEIGHBORS, EMBEDDING_DIM), dtype=np.float32))
    MEM_BANK_PROTOTYPES = tf.constant(np.zeros((NUM_NEIGHBORS, PROTOTYPE_DIM), dtype=np.float32))

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
    Y_candidates_proto = Y_train_val_proto[candidate_indices] # Use Prototypes
    source_ids_candidates = source_ids_train_val[candidate_indices]

    print(f"Z_pool: {n_total} | Z_val (20%): {n_val} | Z_candidates (80%): {len(Z_candidates)}")

    # Step 3: Group by Label
    print("Grouping candidates by label...")
    label_groups = {}
    for i in range(NUM_CLASSES):
        mask = (Y_candidates_int == i)
        label_groups[i] = {
            'vecs': Z_candidates[mask],
            'labels_int': Y_candidates_int[mask],
            'labels_proto': Y_candidates_proto[mask], # Use Prototypes
            'source_ids': source_ids_candidates[mask]
        }
        print(f"  Label {i}: {len(Z_candidates[mask])} candidates")

    current_ltm_vecs = []  # List of arrays
    current_ltm_protos = []  # List of arrays (Changed from labels)
    
    # Load existing vectors if persistent
    if LTM_EXISTS and existing_count > 0:
        current_ltm_vecs = [db_vecs_raw]
        current_ltm_protos = [db_protos_raw]
        print(f"Starting with {existing_count} existing LTM vectors")
    
    best_acc = 0.0
    global_insert_count = 0

    # ---------------------------------------------------------
    # *** FIXED: k-NN Accuracy (Handles Prototypes) ***
    # ---------------------------------------------------------
    def knn_accuracy(query_zs, query_labels, memory_zs, memory_protos, k=5, 
                    hqe_model=None, raw_images=None, temperature=1.0, num_classes=10):
        """
        Matches MultiHopHyperRetriever Memory Bank Retrieval EXACTLY:
        tf.math.top_k → Temperature → tf.nn.softmax → Weighted Prototype Sum
        Then decode prototype to class for accuracy.
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
                # Decode Prototype Output to Class
                for vec in out.numpy():
                    sims = np.dot(assigned_prototypes, vec)
                    preds.append(np.argmax(sims))
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
        
        if isinstance(memory_protos, list):
            memory_protos_arr = np.vstack(memory_protos)
        else:
            memory_protos_arr = memory_protos
        memory_protos_tf = tf.convert_to_tensor(memory_protos_arr, dtype=tf.float32)
        
        # === Compute Cosine Similarity Matrix (EXACT MATCH) ===
        sim_matrix = tf.matmul(query_zs_tf, memory_zs_tf, transpose_b=True)
        
        # === Top-K Selection (EXACT MATCH - tf.math.top_k) ===
        values_main, indices_main = tf.math.top_k(sim_matrix, k=k)
        
        # === Temperature Scaling (EXACT MATCH) ===
        scaled_values_main = values_main / temperature
        
        # === Softmax Attention (EXACT MATCH - tf.nn.softmax) ===
        attn_weights_main = tf.nn.softmax(scaled_values_main, axis=1)
        
        # === Gather Neighbor Prototypes (EXACT MATCH - tf.gather) ===
        neighbor_protos_main = tf.gather(memory_protos_tf, indices_main)
        
        # === Weighted Prototype Sum (EXACT MATCH - tf.reduce_sum) ===
        pred_main = tf.reduce_sum(tf.expand_dims(attn_weights_main, -1) * neighbor_protos_main, axis=1)
        
        # === Decode Prototypes to Classes ===
        pred_classes = []
        for vec in pred_main.numpy():
            sims = np.dot(assigned_prototypes, vec)
            pred_classes.append(np.argmax(sims))
        
        return accuracy_score(query_labels, pred_classes)

    print(f"DEBUG: MEM_BANK_VECS shape: {MEM_BANK_VECS.shape}")
    print(f"DEBUG: current_ltm_vecs total: {sum(len(v) for v in current_ltm_vecs)}")
    print(f"DEBUG: LTM_USE_HQE_FOR_RETRIEVAL: {LTM_USE_HQE_FOR_RETRIEVAL}")
    print(f"DEBUG: HQE_MODEL_AVAILABLE: {HQE_MODEL_AVAILABLE}")

    # Step 6: Collect ALL Batches First (Grouped by Label, Then Shuffled)
    print("\nCollecting and Shuffling Batches Across All Labels...")
    all_batches = []

    for label in range(NUM_CLASSES):
        group = label_groups[label]
        group_vecs = group['vecs']
        group_labels_int = group['labels_int']
        group_labels_proto = group['labels_proto']
        group_source_ids = group['source_ids']
        
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
            eligible_labels_proto = group_labels_proto[eligible_mask]
            eligible_source_ids = group_source_ids[eligible_mask]
            
            if len(eligible_vecs) == 0:
                break
            
            # === Sort Ascending by Similarity (Lowest First) ===
            sort_idx = np.argsort(eligible_sims)
            eligible_vecs = eligible_vecs[sort_idx]
            eligible_sims = eligible_sims[sort_idx]
            eligible_labels_int = eligible_labels_int[sort_idx]
            eligible_labels_proto = eligible_labels_proto[sort_idx]
            eligible_source_ids = eligible_source_ids[sort_idx]
            
            # === Batch (Size ie 64) ===
            batch_size = min(LTM_INSERT_BATCH_SIZE, len(eligible_vecs))
            batch_vecs = eligible_vecs[:batch_size]
            batch_labels_int = eligible_labels_int[:batch_size]
            batch_labels_proto = eligible_labels_proto[:batch_size]
            batch_source_ids = eligible_source_ids[:batch_size]
            batch_sims = eligible_sims[:batch_size]
            
            # Store batch for later shuffled processing
            all_batches.append({
                'vecs': batch_vecs,
                'labels_int': batch_labels_int,
                'labels_proto': batch_labels_proto,
                'source_ids': batch_source_ids,
                'sims': batch_sims,
                'label': label
            })
            
            # Remove processed from pool
            group_vecs = eligible_vecs[batch_size:]
            group_labels_int = eligible_labels_int[batch_size:]
            group_labels_proto = eligible_labels_proto[batch_size:]
            group_source_ids = eligible_source_ids[batch_size:]

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
        batch_labels_proto = batch_data['labels_proto']
        batch_source_ids = batch_data['source_ids']
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
            batch_labels_proto = batch_labels_proto[keep_mask]
            batch_source_ids = batch_source_ids[keep_mask]
            
            if len(batch_vecs) == 0:
                continue
        
        # === Test & Accept ===
        # Temp Add
        temp_ltm_vecs = current_ltm_vecs + [batch_vecs]
        temp_ltm_vecs_arr = np.vstack(temp_ltm_vecs)
        temp_ltm_protos = current_ltm_protos + [batch_labels_proto]
        temp_ltm_protos_arr = np.vstack(temp_ltm_protos)

        # === Validate on ALL Z_val (mixed classes) ===
        # *** FIXED: Validation queries should match training/inference retrieval ***
        if LTM_USE_HQE_FOR_RETRIEVAL and HQE_MODEL_AVAILABLE and hqe_model_for_encoding is not None:
            #use hqe model
            acc = knn_accuracy(
                None,
                Y_val_int, 
                temp_ltm_vecs_arr, 
                temp_ltm_protos_arr, 
                k=NUM_NEIGHBORS,
                hqe_model=hqe_model_for_encoding,
                raw_images=X_train_val[val_indices],
                num_classes=NUM_CLASSES
            )
            print(f"  Validation: HQE queries vs {'HQE' if USE_HQE_ENCODING else 'Frozen'} memory")
        else:
            acc = knn_accuracy(
                Z_val,
                Y_val_int, 
                temp_ltm_vecs_arr, 
                temp_ltm_protos_arr, 
                k=NUM_NEIGHBORS,
                num_classes=NUM_CLASSES
            )
            print(f"  Validation: Frozen queries vs Frozen memory")
        
        if acc > best_acc:
            # ACCEPT
            best_acc = acc
            current_ltm_vecs.append(batch_vecs)
            current_ltm_protos.append(batch_labels_proto)
            
            # *** FIXED: Make room BEFORE inserting (FIFO eviction) ***
            make_room_for_insert(collection, len(batch_vecs), LTM_MAX_CAPACITY, "LTM")
            
            # *** FIXED: Generate unique IDs with timestamp to avoid duplicates ***
            current_timestamp = datetime.now().timestamp()
            ids_to_insert = []
            metadatas_to_insert = []
            for idx in range(len(batch_vecs)):
                unique_id = generate_unique_id("ltm_seed", global_insert_count + idx, current_timestamp)
                ids_to_insert.append(unique_id)
                
                metadatas_to_insert.append({
                    "true_label": int(batch_labels_int[idx]),
                    "source_label": int(batch_labels_int[idx]),
                    "source_image_id": int(batch_source_ids[idx]),
                    "prototype_vector": json.dumps(batch_labels_proto[idx].tolist()),  # JSON String
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
            current_ltm_protos = []
            for m in results['metadatas']:
                current_ltm_protos.append(json.loads(m['prototype_vector']))  # Parse JSON string
            current_ltm_protos = [np.array(current_ltm_protos).astype('float32')]  # Now convert to array


            # Update MEM_BANK_VECS for next validation iteration
            MEM_BANK_VECS = tf.constant(current_ltm_vecs[0])
            MEM_BANK_PROTOTYPES = tf.constant(current_ltm_protos[0])
            
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

db_protos_raw = []
for m in results['metadatas']:
    try: 
        db_protos_raw.append(json.loads(m['prototype_vector']))  # Parse JSON string
    except: 
        db_protos_raw.append([0]*PROTOTYPE_DIM) 
        
db_protos_raw = np.array(db_protos_raw).astype('float32')
MEM_BANK_VECS = tf.constant(db_vecs_raw)
MEM_BANK_PROTOTYPES = tf.constant(db_protos_raw)

print(f"LTM Loaded: {len(db_vecs_raw)} vectors (Max: {LTM_MAX_CAPACITY})")

# ---------------------------------------------------------
# 8. Compile
# ---------------------------------------------------------
system_model = GuidedSystem(retriever_branch)

# ---------------------------------------------------------
# *** NEW: Custom Prototype Accuracy Metric ***
# ---------------------------------------------------------
class PrototypeAccuracy(tf.keras.metrics.Metric):
    def __init__(self, assigned_prototypes, name='prototype_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.assigned_prototypes = tf.constant(assigned_prototypes, dtype=tf.float32)
        self.num_classes = len(assigned_prototypes)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.correct = self.add_weight(name='correct', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: 128-dim prototype vectors (training targets)
        # y_pred: 128-dim model output
        
        # Find which class each y_true belongs to (by matching to assigned_prototypes)
        true_sims = tf.matmul(y_true, self.assigned_prototypes, transpose_b=True)
        true_classes = tf.argmax(true_sims, axis=1)
        
        # Find predicted class (1-NN to output)
        pred_sims = tf.matmul(y_pred, self.assigned_prototypes, transpose_b=True)
        pred_classes = tf.argmax(pred_sims, axis=1)
        
        # Compare
        matches = tf.cast(tf.equal(true_classes, pred_classes), tf.float32)
        
        if sample_weight is not None:
            matches = matches * sample_weight
            self.total.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.total.assign_add(tf.cast(tf.shape(matches)[0], tf.float32))
        
        self.correct.assign_add(tf.reduce_sum(matches))
    
    def result(self):
        return self.correct / (self.total + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.total.assign(0)
        self.correct.assign(0)

# Create metric instance (after assigned_prototypes is defined)
proto_accuracy_metric = PrototypeAccuracy(assigned_prototypes)

# ---------------------------------------------------------
# *** NEW: PROTOTYPE ALIGNMENT LOSS ***
# ---------------------------------------------------------
def prototype_alignment_loss(y_true, y_pred):
    """
    y_true: Target Prototype Vector (PROTOTYPE_DIM)
    y_pred: Model Output Vector (PROTOTYPE_DIM)
    Loss: 1 - Cosine Similarity (Minimize distance on hypersphere)
    """
    # Ensure normalized
    y_true_norm = tf.nn.l2_normalize(y_true, axis=1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=1)
    
    # Cosine Similarity
    similarity = tf.reduce_sum(y_true_norm * y_pred_norm, axis=1)
    
    # Loss (Minimize distance -> Maximize Similarity)
    loss = 1.0 - similarity
    
    return tf.reduce_mean(loss)

# *** USE LOADED OPTIMIZER IF AVAILABLE ***
if loaded_optimizer is not None and MODEL_LOADED:
    print(f"\nUsing loaded optimizer (LR = {loaded_lr:.9f})")
    system_model.compile(
        optimizer=loaded_optimizer,  # ← Use loaded optimizer
        loss=prototype_alignment_loss, # ← Use Prototype Loss
        metrics=[proto_accuracy_metric] # Metrics handled manually via decoding
    )
else:
    print(f"\nUsing fresh optimizer (LR = {loaded_lr:.9f})")
    system_model.compile(
        optimizer=Adam(learning_rate=loaded_lr),  # ← Use loaded LR
        loss=prototype_alignment_loss, # ← Use Prototype Loss
        metrics=[proto_accuracy_metric]
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
    X_train_val, Y_train_val_proto, # ← Use Prototype Targets
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_te, Y_te_proto), # ← Use Prototype Targets
    callbacks=[early_stop, reduce_lr, TemperatureLogger(), EDACallback()]
)

# ---------------------------------------------------------
# 10. EVALUATION & STM OPTIMIZATION (From Script A - 3 Pass System)
# ---------------------------------------------------------
print("\n_______________________________________________________________________")
print("Evaluation Results (Three-Pass System)")
print("_______________________________________________________________________")

def calculate_accuracy_with_stm(model, X_subset, y_true_subset, stm_vecs_np, stm_protos_np):
    if len(stm_vecs_np) > 0:
        stm_v_tf = tf.constant(stm_vecs_np, dtype=tf.float32)
        stm_p_tf = tf.constant(stm_protos_np, dtype=tf.float32)
    else:
        stm_v_tf = None
        stm_p_tf = None
        
    preds = []
    batch_size = 256
    for i in range(0, len(X_subset), batch_size):
        x_b = X_subset[i:i+batch_size]
        out = model.retriever(x_b, training=False, stm_vecs=stm_v_tf, stm_protos=stm_p_tf)
        # Decode Prototype Output to Class
        for vec in out.numpy():
            sims = np.dot(assigned_prototypes, vec)
            preds.append(np.argmax(sims))
    
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

eval_dataset = tf.data.Dataset.from_tensor_slices((X_te, y_te_int, Y_te_proto)).batch(BATCH_SIZE)

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

for step, (x_batch, y_true_int, y_true_proto) in enumerate(eval_dataset):
    output = system_model(x_batch, training=False, stm_vecs=None, stm_protos=None, return_sim=True)
    pred_final = output['predictions']
    max_sim    = output['max_similarity'].numpy()

    # Decode Prototype Output to Class
    y_pred_cls = []
    for vec in pred_final.numpy():
        sims = np.dot(assigned_prototypes, vec)
        y_pred_cls.append(np.argmax(sims))
    y_pred_cls = np.array(y_pred_cls)
    
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
    y_proto_wrong = y_true_proto.numpy()[wrong_idx]
    y_int_wrong = y_true_int.numpy()[wrong_idx]
    sim_wrong   = max_sim[wrong_idx]

    # STRATEGY 1: Wrong + Low Similarity - Negative Incorrect Example (Repulsor)
    if HYBRID_USE_LOW_SIM:
        is_low_sim = (sim_wrong <= STM_SIMILARITY_THRESHOLD_CAND)
        low_sim_idx = np.where(is_low_sim)[0]
        
        if len(low_sim_idx) > 0:
            for r in low_sim_idx:
                wrong_pred_cls = y_pred_cls[wrong_idx][r]  # 1. Get the incorrect prediction index (e.g., Class 2)
                
                # 2. Get the Prototype Vector for the WRONG class
                wrong_class_proto = assigned_prototypes[wrong_pred_cls]
                
                # 3. Negate it to create a Repulsor
                neg_proto = wrong_class_proto * -1.0 * NEGATIVE_AVOIDANCE_WEIGHT
                
                strategy1_candidates.append({
                    'type': 'low_sim', 
                    'image': x_wrong[r], 
                    'label_proto': neg_proto,  # <--- Stores the Negative Wrong Prototype
                    'label_int': int(y_int_wrong[r]), 
                    'sim': float(sim_wrong[r])
                })

            n_errors_low_sim += len(low_sim_idx)

    # STRATEGY 2: Wrong + Found Correct LTM Prototype - Positive Correct Example
    if HYBRID_USE_LTM_PROTO:
        if (STM_STORE_Q_NOT_Z_STRAT2):
            _, _, z_query, _ = system_model.retriever(
                x_wrong, 
                training=False, 
                stm_vecs=None, 
                stm_protos=None, 
                return_intermediate=True
            )
        else:
            z_query = frozen_enc_layer(x_wrong, training=False)

        z_query_norm = tf.nn.l2_normalize(z_query, axis=1)
        y_proto_wrong_tf = tf.constant(y_proto_wrong, dtype=tf.float32)

        sim_matrix = tf.matmul(z_query_norm, LTM_VECS_NORM, transpose_b=True)
        # Match based on Prototype Similarity
        # We compare the query's retrieved prototype sum against the target prototype
        # Simplified: Check if any LTM vec has high sim AND matches class
        label_match = tf.matmul(y_proto_wrong_tf, MEM_BANK_PROTOTYPES, transpose_b=True)
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
candidate_protos = []
candidate_labels_int = []
candidate_sims       = []
candidate_counts     = []
candidate_sources    = []

if HYBRID_USE_LOW_SIM and len(strategy1_candidates) > 0:
    print(f"Encoding {len(strategy1_candidates)} Strategy 1 candidates...")
    strat1_imgs = np.array([c['image'] for c in strategy1_candidates])
    strat1_protos = np.array([c['label_proto'] for c in strategy1_candidates])
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
                stm_protos=None, 
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
        candidate_protos.append(strat1_protos[j])
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
    g_protos = tf.gather(MEM_BANK_PROTOTYPES, hit_indices).numpy()

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
        candidate_protos.append(g_protos[j])
        candidate_labels_int.append(int(rec['label_int']))
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
        candidate_protos = [candidate_protos[i] for i in unique_indices]
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
    cand_protos = np.array(candidate_protos)
    cand_labels_int = np.array(candidate_labels_int)
    cand_sims = np.array(candidate_sims)
    
    unique_labels = np.unique(cand_labels_int)
    
    # === COLLECT ALL BATCHES FIRST (Grouped by Label) ===
    print(f"Organizing {len(cand_vecs)} candidates into batches by label...")
    all_stm_batches = []
    
    for label in unique_labels:
        mask = (cand_labels_int == label)
        l_vecs = cand_vecs[mask]
        l_protos = cand_protos[mask]
        l_ints = cand_labels_int[mask]
        l_sims = cand_sims[mask]
        
        l_sort_idx = np.argsort(l_sims)
        l_vecs = l_vecs[l_sort_idx]
        l_protos = l_protos[l_sort_idx]
        l_ints = l_ints[l_sort_idx]
        l_sims = l_sims[l_sort_idx]
        
        num_label_batches = int(np.ceil(len(l_vecs) / STM_INSERT_BATCH_SIZE))
        for i in range(num_label_batches):
            start = i * STM_INSERT_BATCH_SIZE
            end = min(start + STM_INSERT_BATCH_SIZE, len(l_vecs))
            
            all_stm_batches.append({
                'vecs': l_vecs[start:end],
                'protos': l_protos[start:end],
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
        out = system_model.retriever(x_b, training=False, stm_vecs=None, stm_protos=None)
        for vec in out.numpy():
            sims = np.dot(assigned_prototypes, vec)
            opt_preds.append(np.argmax(sims))
    
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
    current_stm_protos = []
    
    # Load existing STM if persistent
    if PERSIST_STM_ACROSS_RUNS and existing_stm_count > 0:
        print("Loading existing STM vectors...")
        stm_results = stm_collection.get(include=['embeddings', 'metadatas'])
        current_stm_vecs = [np.array(stm_results['embeddings']).astype('float32')]
        current_stm_protos = []
        for m in stm_results['metadatas']:
            try:
                current_stm_protos.append(json.loads(m['prototype_vector']))  # Parse JSON string
            except:
                current_stm_protos.append([0]*PROTOTYPE_DIM)

        current_stm_protos = [np.array(current_stm_protos).astype('float32')]
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
        batch_protos = batch_data['protos']
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
            batch_protos = batch_protos[keep_mask]
            batch_labels_int = batch_labels_int[keep_mask]
            
            if len(batch_vecs) == 0:
                print(f"  Batch {batch_idx+1} (Class {batch_label}): SKIPPED (Empty After Filter)")
                batch_idx += 1
                continue
        
        # === Test & Accept ===
        temp_stm_vecs = current_stm_vecs + [batch_vecs] if current_stm_vecs else [batch_vecs]
        temp_stm_vecs_np = np.vstack(temp_stm_vecs)
        temp_stm_protos_np = np.vstack(current_stm_protos + [batch_protos]) if current_stm_protos else batch_protos
        
        temp_acc_test = calculate_accuracy_with_stm(system_model, X_opt_errors, y_opt_errors, temp_stm_vecs_np, temp_stm_protos_np)
        temp_acc_train = calculate_accuracy_with_stm(system_model, X_opt2, y_opt_int2, temp_stm_vecs_np, temp_stm_protos_np)
        
        greater_than_zero = temp_acc_test > 0.000
        test_rate_improved = temp_acc_test >= best_acc - 0.0001
        train_rate_improved = temp_acc_train >= temp_acc_train_1 - 0.0001
        
        print(f"  Test: {temp_acc_test:.4f} Train: {temp_acc_train:.4f}")

        if total_inserted < STM_BOOTSTRAP_TOTAL or (greater_than_zero and np.logical_and(test_rate_improved, train_rate_improved).all()):
            best_acc = max(temp_acc_test, best_acc)
            current_stm_vecs.append(batch_vecs)
            current_stm_protos.append(batch_protos)
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
                
                metadatas_to_insert.append({
                    "true_label": int(batch_labels_int[idx]), 
                    "prototype_vector": json.dumps(batch_protos[idx].tolist()),  # JSON String
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
            current_stm_protos = []
            for m in stm_results['metadatas']:
                current_stm_protos.append(json.loads(m['prototype_vector']))  # json load string
            current_stm_protos = [np.array(current_stm_protos).astype('float32')]
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
    stm_protos_final = []
    for m in stm_results['metadatas']:
        stm_protos_final.append(json.loads(m['prototype_vector']))  # json load string
    stm_protos_final = np.array(stm_protos_final).astype('float32')

    # *** VACUUM STM DATABASE AFTER OPTIMIZATION ***
    vacuum_chroma_database(STM_DB_PATH)
    
else:
    stm_vecs_final = np.empty((0, EMBEDDING_DIM))
    stm_protos_final = np.empty((0, PROTOTYPE_DIM))
    print(">>> No STM Optimization Performed.")

# === STM FINAL STATE DEBUG ===
print(f"\n========================================")
print(f"STM FINAL STATE CHECK")
print(f"========================================")

# Load directly from ChromaDB (not in-memory lists)
stm_results = stm_collection.get(include=['embeddings', 'metadatas'])
stm_vecs_db = np.array(stm_results['embeddings']).astype('float32')
stm_protos_db = []
for m in stm_results['metadatas']:
    stm_protos_db.append(json.loads(m['prototype_vector']))  # load json string
stm_protos_db = np.array(stm_protos_db).astype('float32')

print(f"STM vectors in ChromaDB: {len(stm_vecs_db)}")
print(f"STM capacity: {STM_MAX_CAPACITY}")
print(f"Utilization: {len(stm_vecs_db)/STM_MAX_CAPACITY*100:.1f}%")

stm_results_not_zero = (len(stm_vecs_db)/STM_MAX_CAPACITY*100) > 1

# Class distribution
if stm_results_not_zero:
    unique, counts = np.unique(np.argmax(stm_protos_db, axis=1), return_counts=True) # Approximate class via proto index if needed, but here we store true_label in metadata usually. For now just count.
    print(f"\nClass Distribution:")
    # We don't have direct class index in proto vector, but we can check metadata if needed. 
    # For brevity, skipping detailed class dist print unless metadata is parsed again.

# Test accuracy with DB state (not in-memory)
print(f"\nAccuracy Test:")
test_acc = calculate_accuracy_with_stm(
    system_model, X_opt_errors, y_opt_errors,
    stm_vecs_db, stm_protos_db
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

eval_dataset = tf.data.Dataset.from_tensor_slices((X_te, y_te_int, Y_te_proto)).batch(BATCH_SIZE)

pass3_preds = []
pass3_trues = []

stm_v_tf = tf.constant(stm_vecs_final, dtype=tf.float32) if len(stm_vecs_final) > 0 else None
stm_p_tf = tf.constant(stm_protos_final, dtype=tf.float32) if len(stm_protos_final) > 0 else None

for step, (x_batch, y_true_int, y_true_proto) in enumerate(eval_dataset):
    output = system_model(x_batch, training=False, stm_vecs=stm_v_tf, stm_protos=stm_p_tf, return_sim=False)
    # Decode Prototype Output to Class
    batch_preds = []
    for vec in output.numpy():
        sims = np.dot(assigned_prototypes, vec)
        batch_preds.append(np.argmax(sims))
    
    pass3_preds.extend(batch_preds)
    pass3_trues.extend(y_true_int.numpy())

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
# *** VALUE ENCODER BASELINE REMOVED FOR DPAD COMPATIBILITY ***
print(f"\nFinal Accuracy Comparison:")
print(f"Multi-Hop Hyper Pass 1 (No STM): {acc_pass1:.4f}")
print(f"Multi-Hop Hyper Pass 3 (Opt STM): {acc_pass3:.4f}")

if acc_pass3 > acc_pass1:
    print(f"\n*** OPTIMIZED STM IMPROVEMENT: +{(acc_pass3 - acc_pass1)*100:.2f}% ***")
    if HYPERPARM_GRID_SEARCH:
        temp_improvement = (acc_pass3 - acc_pass1)*100
        temp_best_accuracy_saved = load_variable(GRID_SEARCH_BEST_ACCURACY_FILE_PATH, 0)
        if temp_improvement > temp_best_accuracy_saved:
            save_variable(GRID_SEARCH_BEST_ACCURACY_FILE_PATH, temp_improvement)
            save_variable(GRID_SEARCH_BEST_INDEX_FILE_PATH, GRID_SEARCH_INDEX)
            print("\n_______________________________________________________________________")
            print(f"\n*** ACCURACY SAVED AT INDEX: {GRID_SEARCH_INDEX} ***")
            print("_______________________________________________________________________")


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
print(f"Prototype Vectors saved to: {PROTOTYPE_SAVE_PATH}")
print(f"Prototype LUT saved to: {PROTOTYPE_LUT_PATH}")

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
print(f"Prototype Vectors Saved: {PROTOTYPE_SAVE_PATH}")
print(f"Prototype LUT Saved: {PROTOTYPE_LUT_PATH}")
print(f"\nNext run will:")
print(f"  - Load existing LTM ({final_ltm_count} vectors)")
print(f"  - Load existing STM ({final_stm_count} vectors)")
print(f"  - Load previous model from .keras + config fallback")
print(f"  - Load previous optimizer state (with learning rate)")
print(f"  - Load existing visual centroids ({NUM_VISUAL_CENTROIDS} centroids)")
print(f"  - Load existing hyperspherical prototypes ({PROTOTYPE_COUNT} prototypes)")
print(f"  - Load existing prototype LUT (Dataset -> Class -> Prototype)")
print(f"  - Use HQE model for LTM encoding (if weights exist)")
print(f"  - Append new vectors with FIFO eviction (continuous learning)")
print(f"\n*** ABLATION STUDY FLAGS ***")
print(f"  - LTM_USE_FROZEN_ENCODER_FOR_INSERTION: {LTM_USE_FROZEN_ENCODER_FOR_INSERTION}")
print(f"  - LTM_USE_HQE_FOR_RETRIEVAL: {LTM_USE_HQE_FOR_RETRIEVAL}")
print(f"\n*** DPAD BRANCHES ***")
print(f"  - QE Branch: ENABLED")
print(f"  - VE Branch: {USE_VE_BRANCH}")
print(f"  - DE Branch: {USE_DE_BRANCH}")
print(f"\n*** LEARNING RATE ***")
print(f"  - Current LR: {system_model.optimizer.learning_rate.numpy():.9f}")
print(f"  - Will persist to next run: YES")

print("\n_______________________________________________________________________")
print("Training Complete!")
print("_______________________________________________________________________")