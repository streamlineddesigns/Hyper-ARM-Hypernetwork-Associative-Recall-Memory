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
# SQLITE FIX
# ---------------------------------------------------------
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    print("ERROR: pysqlite3-binary not installed!")
    exit()

# ---------------------------------------------------------
# INFERENCE & VALIDATION SCRIPT WITH TOP-K SAMPLING
# ---------------------------------------------------------

import numpy as np
import ast
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
import chromadb

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MAIN_DB_PATH = "./chroma_db_mnist"
MAIN_COLLECTION_NAME = "mnist_sparse_collection"
CENTROID_DB_PATH = "./chroma_db_centroids"
CENTROID_COLLECTION_NAME = "mnist_cluster_centers"

ENCODER_PATH = "./saved_cnne_model_dir"
VALUE_ENC_PATH = "./saved_mnist_classifier_dir" 
HQE_MODEL_PATH = "./saved_qe_hyper_system_optimized"

EMBEDDING_DIM = 392        
NUM_NEIGHBORS = 5          # How many Memory Bank neighbors to check
INFERENCE_TOP_K = 4        # <-- NEW CONFIG: How many Centroid Hypernetworks to ensemble

# --- NETWORK ARCHITECTURE (MUST MATCH TRAINING) ---
TARGET_NET_ARCH = [25]

# ---------------------------------------------------------
# 1. DEFINE ARCHITECTURE CLASSES (Required for Model Loading)
# ---------------------------------------------------------
# Keras needs these definitions to reconstruct the model graph from disk

class FrozenEncoderLayer(layers.Layer):
    def __init__(self, module, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self.trainable = False
    def call(self, inputs):
        res = self.module(inputs)
        return tf.reshape(res, [tf.shape(res)[0], -1]) 
    def get_config(self):
        return super().get_config()

class QueryPreprocessor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim_arg = output_dim # Save for config
        self.cnn_stack = keras.Sequential([
            layers.Conv2D(16, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(output_dim, activation='linear') 
        ])
    def call(self, x_raw):
        return self.cnn_stack(x_raw)
    def get_config(self):
        return {"output_dim": self.output_dim_arg}

class QueryEncoder(keras.Model):
    def __init__(self, output_param_count, **kwargs):
        super().__init__(**kwargs)
        self.op_count = output_param_count
        self.net = keras.Sequential([
            layers.Dense(98, activation='relu'), # Hardcoded intermediate dim from training
            layers.Dense(output_param_count, activation='linear')
        ])
    def call(self, centroid_context):
        return self.net(centroid_context)
    def get_config(self):
        return {"output_param_count": self.op_count}

class DynamicTargetNetwork(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, z_prime_q_input, generated_params):
        current_x = z_prime_q_input
        current_dim = tf.shape(current_x)[-1]
        offset = 0
        
        # Hidden Layers
        for i, next_dim in enumerate(TARGET_NET_ARCH):
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
            
        # Output Layer
        out_dim = EMBEDDING_DIM 
        w_size_out = current_dim * out_dim
        b_size_out = out_dim
        
        w_flat_out = generated_params[:, offset : offset + w_size_out]
        b_val_out  = generated_params[:, offset + w_size_out : offset + w_size_out + b_size_out]
        
        w_matrix_out = tf.reshape(w_flat_out, [tf.shape(current_x)[0], current_dim, out_dim])
        z_prime_delta = tf.einsum('bi,bij->bj', current_x, w_matrix_out) + b_val_out
        
        return z_prime_delta

class HyperRetrieverSystem(Model):
    """
    Modified Architecture Definition to allow 'training=False' usage safely
    """
    def __init__(self, enc_layer, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.enc_layer_conf = enc_layer # placeholder
        self.out_dim_conf = output_dim
        
        self.encoder = enc_layer
        self.preprocessor = QueryPreprocessor(output_dim)
        self.qe = QueryEncoder(self._calc_params(output_dim))
        self.target_net = DynamicTargetNetwork()
        
    def _calc_params(self, inp_dim):
        count = 0; prev = inp_dim
        for size in TARGET_NET_ARCH:
            count += prev * size + size; prev = size
        count += prev * inp_dim + inp_dim
        return count

    # Standard call uses Argmax (Training Mode logic)
    def call(self, inputs, training=None):
        z = self.encoder(inputs)
        z_norm = tf.linalg.l2_normalize(z, axis=1)
        c_norm = tf.linalg.l2_normalize(CENTROID_VECS_TF, axis=1)
        sims = tf.matmul(z_norm, c_norm, transpose_b=True)
        best_idx = tf.argmax(sims, axis=-1)
        ctx_vec = tf.gather(CENTROID_VECS_TF, best_idx)
        
        gen_params = self.qe(ctx_vec)
        z_prime_q = self.preprocessor(inputs)
        z_prime_delta = self.target_net(z_prime_q, gen_params)
        q = z + z_prime_delta
        
        q_norm = tf.linalg.l2_normalize(q, axis=1)
        mem_sim = tf.matmul(q_norm, MEM_BANK_TF, transpose_b=True)
        values, indices = tf.math.top_k(mem_sim, k=NUM_NEIGHBORS)
        attn_weights = tf.nn.softmax(values, axis=1)
        neighbor_labels = tf.gather(MEM_LABELS_TF, indices)
        pred = tf.reduce_sum(tf.expand_dims(attn_weights, -1) * neighbor_labels, axis=1)
        return pred
    
    def get_config(self):
        return {"enc_layer": self.enc_layer_conf, "output_dim": self.out_dim_conf}

# ---------------------------------------------------------
# 2. LOAD ENVIRONMENT & DATABASES
# ---------------------------------------------------------

print("--- INITIALIZING VALIDATION SYSTEM ---")

# Fix sqlite
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

print("1. Connecting Databases...")
client_main = chromadb.PersistentClient(path=MAIN_DB_PATH)
collection_main = client_main.get_collection(MAIN_COLLECTION_NAME)
res_main = collection_main.get(include=['embeddings', 'metadatas'])

MEM_BANK_NP = np.array(res_main['embeddings']).astype('float32')
MEM_BANK_TF = tf.constant(MEM_BANK_NP)

MEM_LABELS = []
for m in res_main['metadatas']:
    try: MEM_LABELS.append(ast.literal_eval(m['one_hot_vector']))
    except: MEM_LABELS.append([0]*10) 
MEM_LABELS_TF = tf.constant(np.array(MEM_LABELS).astype('float32'))

client_cent = chromadb.PersistentClient(path=CENTROID_DB_PATH)
collection_cent = client_cent.get_collection(CENTROID_COLLECTION_NAME)
res_cent = collection_cent.get(include=['embeddings'])
CENTROID_VECS_NP = np.array(res_cent['embeddings']).astype('float32')
CENTROID_VECS_TF = tf.constant(CENTROID_VECS_NP)

# Globalize variables so the class methods see them (Quick fix for scope in inference)
# Alternatively pass them into call(), but modifying signature breaks saved model loading sometimes.
# We rely on them being accessible here.

print("2. Loading Models...")
# Load Value Encoder
value_enc = models.load_model(VALUE_ENC_PATH)

# Load Frozen AE
frozen_obj = tf.saved_model.load(ENCODER_PATH)
frozen_layer = FrozenEncoderLayer(frozen_obj)

# Load HQE System (Retriever)
# Pass custom_objects map so Keras knows about our classes
retriever_model = models.load_model(HQE_MODEL_PATH, custom_objects={
    'FrozenEncoderLayer': FrozenEncoderLayer,
    'QueryPreprocessor': QueryPreprocessor,
    'QueryEncoder': QueryEncoder,
    'DynamicTargetNetwork': DynamicTargetNetwork,
    'HyperRetrieverSystem': HyperRetrieverSystem
})
retriever_model.trainable = False
value_enc.trainable = False

print("3. Loading Test Data (Fashion MNIST)...")
(_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

# ---------------------------------------------------------
# 3. ADVANCED INFERENCE LOGIC (TOP-K ENSEMBLE)
# ---------------------------------------------------------

def inference_top_k(inputs_tensor, k_val):
    """
    Custom inference loop implementing Top-K Hypernetwork Sampling.
    """
    batch_size = tf.shape(inputs_tensor)[0]
    
    # Step A: Common Feature Extraction
    # These computations happen once regardless of K
    z_static = retriever_model.encoder(inputs_tensor)      # Z (Static Embedding)
    z_dyn_input = retriever_model.preprocessor(inputs_tensor) # Z'q (Dynamic Input to Target Net)
    
    # Step B: Identify Top-K Clusters
    z_s = tf.linalg.l2_normalize(z_static, axis=1)
    c_s = tf.linalg.l2_normalize(CENTROID_VECS_TF, axis=1)
    all_sims = tf.matmul(z_s, c_s, transpose_b=True) # Shape: [Batch, Num_Centroids]
    
    # Get Top-K scores and indices
    topk_scores, topk_indices = tf.math.top_k(all_sims, k=k_val)
    # Convert scores to weights (Softmax across K dimension)
    cent_weights = tf.nn.softmax(topk_scores, axis=1) 
    
    # Initialize accumulation variable
    final_retrieval_pred = tf.zeros([batch_size, 10])
    
    # Step C: Iterate over K clusters and ensemble predictions
    # We process each cluster candidate one by one (Looping in python/graph mixed)
    # tf.map_fn or simple python loop over batch if batch small. 
    # Here we use tensor slicing which is efficient enough for moderate k.
    
    for i in range(k_val):
        # Gather specific context for index i
        idx_i = topk_indices[:, i]       # [Batch]
        ctx_i = tf.gather(CENTROID_VECS_TF, idx_i) # [Batch, Dim]
        wt_i = cent_weights[:, i:i+1]     # [Batch, 1]
        
        # 1. Generate Parameters specific to Cluster i
        params_i = retriever_model.qe(ctx_i)
        
        # 2. Execute Target Network with Z'q (Fixed) and Params_i (Changing)
        delta_i = retriever_model.target_net(z_dyn_input, params_i)
        
        # 3. Form Query Q_i = Z + Delta_i
        q_i = z_static + delta_i
        
        # 4. Memory Search for Q_i
        q_i_norm = tf.linalg.l2_normalize(q_i, axis=1)
        mem_sims_i = tf.matmul(q_i_norm, MEM_BANK_TF, transpose_b=True)
        
        vals_i, inds_i = tf.math.top_k(mem_sims_i, k=NUM_NEIGHBORS)
        attn_i = tf.nn.softmax(vals_i, axis=1)
        nbr_labels_i = tf.gather(MEM_LABELS_TF, inds_i)
        
        # Prediction for this specific Hypernetwork instance
        pred_i = tf.reduce_sum(tf.expand_dims(attn_i, -1) * nbr_labels_i, axis=1)
        
        # Accumulate weighted prediction
        final_retrieval_pred += (wt_i * pred_i)
        
    # Step D: Combine with Value Encoder
    p_ve = value_enc(inputs_tensor)
    
    # Final Fusion Formula
    final_output = (final_retrieval_pred + p_ve) / 2.0
    
    return final_output

# ---------------------------------------------------------
# 4. RUN VALIDATION
# ---------------------------------------------------------

# Create wrapper model for easy .predict() if desired, or just loop manually
class TopKValidator(Model):
    def __init__(self, r_model, ve_model, k, **kwargs):
        super().__init__(**kwargs)
        self.r_model = r_model
        self.ve_model = ve_model
        self.k = k
        
    def call(self, inputs):
        return inference_top_k(inputs, self.k)

try:
    validator = TopKValidator(retriever_model, value_enc, INFERENCE_TOP_K)
    
    print(f"\n--- Running Validation (Top-K={INFERENCE_TOP_K}) ---")
    # Note: This inference step effectively runs K-Hypernetworks per image!
    print(f"Expecting {(len(x_test) * INFERENCE_TOP_K)} Hypernetwork Executions...")
    
    y_probs = validator.predict(x_test, batch_size=100, verbose=1)
    y_preds = np.argmax(y_probs, axis=1)
    
    acc = np.mean(y_preds == y_test)
    print(f"\nVALIDATION COMPLETE (Top-K={INFERENCE_TOP_K})")
    print(f"Accuracy: {acc:.4f}")
    
except Exception as e:
    print("\nERROR DURING INFERENCE:")
    import traceback
    traceback.print_exc()