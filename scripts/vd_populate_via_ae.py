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
# NOW continue with your normal imports...
# ---------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import chromadb
from chromadb.config import Settings

# ---------------------------------------------------------
# 1. CONFIGURATION & DATA LOADING
# ---------------------------------------------------------
BATCH_SIZE_PROCESS = 256 # Batch size for predicting embeddings (faster)
SIMILARITY_THRESHOLD = 0.50 # Only insert if similarity is BELOW this value
CHROMA_PATH = "./chroma_db_mnist"
COLLECTION_NAME = "mnist_sparse_collection"

print("Loading MNIST Data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize exactly as the AE was trained
all_images = np.concatenate((x_train, x_test), axis=0)
all_labels = np.concatenate((y_train, y_test), axis=0)

# Reshape and Normalize: (N, 28, 28, 1) range [0, 1]
data_reshaped = all_images.astype('float32') / 255.0
data_reshaped = np.expand_dims(data_reshaped, -1)
total_samples = data_reshaped.shape[0]

print(f"Total Images to Process: {total_samples}")

# ---------------------------------------------------------
# 2. LOAD FROZEN ENCODER
# ---------------------------------------------------------
saved_encoder_path = "./saved_cnne_model_dir"

if not os.path.exists(saved_encoder_path):
    raise FileNotFoundError(f"Encoder not found at {saved_encoder_path}. Run previous steps first.")

print("Loading Frozen Encoder...")
loaded_obj = tf.saved_model.load(saved_encoder_path)

# We need a way to batch-predict using the loaded object efficiently
class FrozenEncoderWrapper(tf.keras.Model):
    def __init__(self, module):
        super(FrozenEncoderWrapper, self).__init__()
        self.module = module

    def call(self, inputs):
        return self.module(inputs)

encoder_model = FrozenEncoderWrapper(loaded_obj)

def get_embedding(images):
    """Passes images through the network and FLATTENS the output."""
    # Predict
    features = encoder_model.predict(images, verbose=0)
    # Flatten the convolutional feature maps into a single vector
    return features.reshape(features.shape[0], -1)

# Quick test run to ensure dimensions
test_emb = get_embedding(data_reshaped[0:1])
EMBEDDING_DIM = test_emb.shape[1]
print(f"Embedding Dimension determined: {EMBEDDING_DIM}")

# ---------------------------------------------------------
# 3. SETUP CHROMADB
# ---------------------------------------------------------
# Remove existing DB if you want a fresh run
# if os.path.exists(CHROMA_PATH): 
#     import shutil; shutil.rmtree(CHROMA_PATH)

client = chromadb.PersistentClient(path=CHROMA_PATH)

# Use 'cosine' distance so we can intuitively map distance to similarity (1 - dist)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# Dictionary to hold insertion counts per class (0-9)
insert_counts = {i: 0 for i in range(10)}

print(f"\nStarting Processing Loop (Threshold: {SIMILARITY_THRESHOLD})...")
print("---------------------------------------------------------")

# ---------------------------------------------------------
# 4. MAIN LOOP: EMBED -> QUERY -> DECIDE -> INSERT
# ---------------------------------------------------------

# Process in chunks to speed up Tensorflow prediction, 
# but loop individually for decision logic (or batch logic where possible)
# For strict per-item logic, we iterate:

# ---------------------------------------------------------
# (Updated for v0.4.22 compatibility)
# ---------------------------------------------------------

starting_point = 0
stopping_point = 25000
for i in range(starting_point, stopping_point):
    #skip 70% of numbers
    if i % 3 == 0:
        continue
    if i % 4 == 0:
        continue
    if i % 5 == 0:
        continue
    if i % 6 == 0:
        continue
    if i % 7 == 0:
        continue
    if i % 8 == 0:
        continue
    if i % 9 == 0:
        continue

    # 1. Prepare Image
    img = data_reshaped[i:i+1]
    true_label = int(all_labels[i])
    
    # 2. Get Embedding
    current_embedding = get_embedding(img)[0] 
    
    doc_id = f"img_{i}"
    
    do_insert = False
    
    # 3. QUERY LOGIC
    if collection.count() == 0:
        do_insert = True
    else:
        # Query for 1 nearest neighbor
        results = collection.query(
            query_embeddings=[current_embedding.tolist()],
            n_results=1,
            include=['distances', 'metadatas'] # Explicitly ask for distances
        )
        
        # CHROMA 0.4.22 COMPATIBILITY CHECK
        # Ensure results actually came back
        if results['distances'] and results['distances'][0]:
            
            min_distance = results['distances'][0][0]
            max_similarity = 1.0 - min_distance
            
            # GATE CHECK
            if max_similarity < SIMILARITY_THRESHOLD:
                do_insert = True
        
        # Handle empty collection case explicitly for safety
        elif not results['ids'] or results['ids'] == [[]]:
             do_insert = True

    # 4. INSERT
    if do_insert:
        ground_truth_vector = [0] * 10
        ground_truth_vector[true_label] = 1
        
        collection.add(
            ids=[doc_id],
            embeddings=[current_embedding.tolist()],
            metadatas=[{
                "true_label": true_label,
                "one_hot_vector": str(ground_truth_vector)
            }]
        )
        insert_counts[true_label] += 1

    if (i + 1) % 1000 == 0 or i == 0:
        print(f"Processed {i + 1}/{total_samples} images. Current DB Size: {collection.count()}")

print("\nProcess Complete.")
for label, count in sorted(insert_counts.items()):
    print(f"Class {label}: {count} inserts")
