#---------------------------------------------------------
#   Copyright 2026 Pierce Prange
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
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    print("ERROR: pysqlite3-binary not installed!")
    exit()

import numpy as np
import chromadb
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
SOURCE_DB_PATH = "./chroma_db_mnist"
SOURCE_COLLECTION_NAME = "mnist_sparse_collection"

TARGET_DB_PATH = "./chroma_db_centroids"
TARGET_COLLECTION_NAME = "mnist_cluster_centers"

NUM_CLUSTERS = 128

# ---------------------------------------------------------
# 2. CONNECT TO SOURCE AND EXTRACT DATA
# ---------------------------------------------------------
print(f"Connecting to Source DB...")
source_client = chromadb.PersistentClient(path=SOURCE_DB_PATH)
source_collection = source_client.get_or_create_collection(name=SOURCE_COLLECTION_NAME)

print(f"Fetching records...")
results = source_collection.get(include=['embeddings', 'metadatas'])

embedding_matrix = np.array(results['embeddings'])
metadata_list = results['metadatas']

print(f"Original Shape: {embedding_matrix.shape}")

# ---------------------------------------------------------
# 3. PREPROCESSING FOR COSINE COMPATIBILITY
# ---------------------------------------------------------

print("Normalizing Vectors (making them compatible with Cosine Distance)...")
normalized_embeddings = normalize(embedding_matrix, norm='l2', axis=1)

# ---------------------------------------------------------
# 4. RUN K-MEANS ON NORMALIZED DATA
# ---------------------------------------------------------
print("Running K-Means...")

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(normalized_embeddings)
raw_centers = kmeans.cluster_centers_

# Normalize centers before saving to DB
final_normalized_centers = normalize(raw_centers, norm='l2', axis=1)

print("Clustering & Normalization Complete.")

# ---------------------------------------------------------
# 5. ANALYZE CLUSTERS (FIXED LOGIC)
# ---------------------------------------------------------
cluster_analysis = {}

for i in range(NUM_CLUSTERS):
    indices_in_this_cluster = np.where(cluster_labels == i)[0]
    
    counts = {}
    
    # FIX APPLIED HERE: Use len() check instead of raw 'if array'
    if len(indices_in_this_cluster) > 0:
        for idx in indices_in_this_cluster:
            lbl = int(metadata_list[idx]['true_label'])
            counts[lbl] = counts.get(lbl, 0) + 1
            
        majority_label = max(counts, key=counts.get)
        confidence = counts[majority_label] / len(indices_in_this_cluster)
        
        cluster_analysis[i] = {
            "dominant_class": majority_label,
            "count": len(indices_in_this_cluster),
            "confidence": round(confidence, 4)
        }
        
        print(f"Cluster {i}: Size={len(indices_in_this_cluster)}, "
              f"Dominant Class={majority_label} ({confidence:.2%})")
    else:
        # Handle empty clusters (unlikely with 191 items & 10 clusters, but good practice)
        cluster_analysis[i] = {
            "dominant_class": -1,
            "count": 0,
            "confidence": 0
        }
        print(f"Cluster {i}: EMPTY")

# ---------------------------------------------------------
# 6. SAVE TO TARGET DB
# ---------------------------------------------------------
print(f"\nSaving Centroids to New Cosine-Compatible DB: {TARGET_DB_PATH}")

target_client = chromadb.PersistentClient(path=TARGET_DB_PATH)
target_collection = target_client.get_or_create_collection(
    name=TARGET_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

ids_to_insert = []
vecs_to_insert = []
metas_to_insert = []

for i in range(NUM_CLUSTERS):
    c_vec = final_normalized_centers[i].tolist()
    info = cluster_analysis[i]
    
    meta = {
        "cluster_id": i,
        "size": info["count"],
        "estimated_digit": info["dominant_class"],
        "purity_score": info["confidence"]
    }
    
    ids_to_insert.append(f"centroid_cosine_{i}")
    vecs_to_insert.append(c_vec)
    metas_to_insert.append(meta)

target_collection.add(
    ids=ids_to_insert,
    embeddings=vecs_to_insert,
    metadatas=metas_to_insert
)

print(f"Success! Saved {target_collection.count()} vectors.")
print("-" * 30)

# ---------------------------------------------------------
# 7. VERIFICATION DEMO
# ---------------------------------------------------------
print("\nVerification Query:")
test_idx = 50 # Arbitrary test index

query_vec_raw = embedding_matrix[test_idx] 
# Normalize query vec so it matches the normalized centroids (COSINE COMPATIBLE!)
query_vec_norm = normalize([query_vec_raw], norm='l2')[0] 

true_lbl = metadata_list[test_idx]['true_label']

res = target_collection.query(
    query_embeddings=[query_vec_norm.tolist()],
    n_results=1,
    include=['distances', 'metadatas']
)

pred_meta = res['metadatas'][0][0]
dist = res['distances'][0][0]

print(f"Query Image Label: {true_lbl}")
print(f"Best Match ID: {pred_meta['cluster_id']}")
print(f"Predicted Digit: {pred_meta['estimated_digit']}")
print(f"Cosine Distance: {dist:.4f} (Lower is closer/better)")