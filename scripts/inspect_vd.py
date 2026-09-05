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

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import chromadb

# ---------------------------------------------------------
# 1. CONFIGURATION & LOADING MODEL
# ---------------------------------------------------------
# Paths must match exactly where they were saved/stored
SAVED_ENCODER_PATH = "./saved_cnne_model_dir"
CHROMA_DB_PATH = "./chroma_db_mnist"
COLLECTION_NAME = "mnist_sparse_collection"

def get_encoder_model():
    """Loads the frozen encoder used originally."""
    loaded_obj = tf.saved_model.load(SAVED_ENCODER_PATH)
    
    class FrozenEncoderWrapper(tf.keras.Model):
        def __init__(self, module):
            super(FrozenEncoderWrapper, self).__init__()
            self.module = module
        def call(self, inputs):
            return self.module(inputs)

    return FrozenEncoderWrapper(loaded_obj)

print("Loading Neural Network...")
encoder = get_encoder_model()

# ---------------------------------------------------------
# 2. CONNECT TO EXISTING DATABASE
# ---------------------------------------------------------
print("Connecting to Database...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Check if collection exists
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"SUCCESS: Found collection '{COLLECTION_NAME}'")
    print(f"         Total Vectors Stored: {collection.count()}")
except Exception as e:
    print(f"ERROR: Could not find database. Did you run the insert script? {e}")
    exit()

# ---------------------------------------------------------
# SCENARIO A: QUERY THE DATABASE (Find Nearest Neighbors)
# ---------------------------------------------------------

def search_similar_images(image_array_28x28, n_results=3):
    """
    Takes a raw 28x28 image array (or batch), encodes it,
    and searches ChromaDB for matches.
    """
    # 1. Preprocess exactly like before
    img_batch = image_array_28x28.astype('float32') / 255.0
    
    # Ensure shape is (N, 28, 28, 1)
    if img_batch.ndim == 2:
        img_batch = np.expand_dims(img_batch, -1)
    img_batch = np.expand_dims(img_batch, 0) # Add batch dim

    # 2. Generate Embedding
    features = encoder.predict(img_batch, verbose=0)
    # Flatten!
    query_vector = features.reshape(1, -1)[0].tolist() 

    # 3. Query Chroma
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=['metadatas', 'distances'] # Retrieve label info too
    )

    return results

# --- LET'S RUN A TEST ---
print("\n--- RUNNING TEST QUERY ---")

# Let's grab one image from MNIST to test (e.g., a digit '7')
(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
test_img = x_train[10] 
test_label = y_train[10]

print(f"Querying with Image (Label: {test_label})...")

results = search_similar_images(test_img, n_results=3)

print("Results Found:")
# Parse results
ids = results['ids'][0]
distances = results['distances'][0]
metadatas = results['metadatas'][0]

for i in range(len(ids)):
    dist = distances[i]
    sim_score = 1 - dist # Convert distance back to similarity
    lbl = metadatas[i]['true_label']
    
    # Determine status
    status = "EXACT MATCH" if sim_score > 0.999 else "SIMILAR"
    
    print(f"  Match {i+1}: ID={ids[i]} | Label={lbl} | Similarity={sim_score:.4f} ({status})")


# ---------------------------------------------------------
# SCENARIO B: INSPECT / DUMP DATA
# ---------------------------------------------------------

print("\n--- INSPECTING RAW DATA ---")

# Get first 5 items from DB
peek = collection.get(limit=5, include=["metadatas", "embeddings"])

print(f"Fetched {len(peek['ids'])} sample records.")
print(f"Example ID: {peek['ids'][0]}")
print(f"Example Metadata: {peek['metadatas'][0]}")
print(f"Example Embedding Vector Length: {len(peek['embeddings'][0])}")

# Count specific classes currently in DB
# Note: Chroma doesn't have native group-by-count SQL, so we do quick python math here if needed
all_meta = collection.get(include=["metadatas"])["metadatas"]
class_counts = {}
for m in all_meta:
    lbl = m['true_label']
    class_counts[lbl] = class_counts.get(lbl, 0) + 1

print("\nCurrent Distribution in DB:")
for k in sorted(class_counts.keys()):
    print(f"  Class {k}: {class_counts[k]} items")