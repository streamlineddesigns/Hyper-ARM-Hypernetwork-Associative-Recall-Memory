# ---------------------------------------------------------
# SQLITE FIX & IMPORTS
# ---------------------------------------------------------
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    print("ERROR: pysqlite3-binary not installed!")
    exit()

import numpy as np
import ast 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
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
SAVE_PATH_HQE_SYSTEM = "./saved_qe_hyper_system_optimized"

EMBEDDING_DIM = 392        
NUM_NEIGHBORS = 5         
EPOCHS = 5             
LEARNING_RATE = 0.001
BATCH_SIZE = 32            

# --- NETWORK CONFIGURATION ---
TARGET_NET_ARCH = [25] 

# ---------------------------------------------------------
# DATA PREPARATION (Unchanged)
# ---------------------------------------------------------
print("Loading Data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_full = np.concatenate((x_train, x_test), axis=0)
Y_full = np.concatenate((y_train, y_test), axis=0)
X_processed = X_full.reshape(X_full.shape[0], 28, 28, 1).astype('float32') / 255.0
Y_onehot = tf.keras.utils.to_categorical(Y_full, 10)

indices = np.arange(len(X_processed))
idx_train, idx_test, _, _ = train_test_split(indices, Y_full, test_size=0.2, stratify=Y_full)
X_tr = X_processed[idx_train]; y_tr_int = Y_full[idx_train]; y_tr_hot = Y_onehot[idx_train]
X_te = X_processed[idx_test]; y_te_int = Y_full[idx_test]; y_te_hot = Y_onehot[idx_test]

# ---------------------------------------------------------
# LOAD DBs & ENCODERS
# ---------------------------------------------------------
print("Connecting Databases...")
client_main = chromadb.PersistentClient(path=MAIN_DB_PATH)
collection_main = client_main.get_or_create_collection(MAIN_COLLECTION_NAME)
results_main = collection_main.get(include=['embeddings', 'metadatas'])
MEM_BANK_VECS = tf.constant(np.array(results_main['embeddings']).astype('float32'))
MEM_BANK_LABELS = []
for m in results_main['metadatas']:
    try: MEM_BANK_LABELS.append(ast.literal_eval(m['one_hot_vector']))
    except: MEM_BANK_LABELS.append([0]*10) 
MEM_BANK_LABELS = tf.constant(np.array(MEM_BANK_LABELS).astype('float32'))

client_cent = chromadb.PersistentClient(path=CENTROID_DB_PATH)
collection_cent = client_cent.get_or_create_collection(name=CENTROID_COLLECTION_NAME)
results_cent = collection_cent.get(include=['embeddings'])
CENTROID_VECS = tf.constant(np.array(results_cent['embeddings']).astype('float32'))

# ---------------------------------------------------------
# ARCHITECTURE HELPERS
# ---------------------------------------------------------

def get_target_params_count(input_dim, arch_list):
    count = 0
    prev = input_dim
    for size in arch_list:
        count += prev * size + size 
        prev = size
    count += prev * input_dim + input_dim # Returns to original dim (EMBEDDING_DIM)
    return count

TOTAL_PARAMS_NEEDED = get_target_params_count(EMBEDDING_DIM, TARGET_NET_ARCH)

# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

class FrozenEncoderLayer(layers.Layer):
    def __init__(self, module, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self.trainable = False
    def call(self, inputs):
        res = self.module(inputs)
        return tf.reshape(res, [tf.shape(res)[0], -1]) 

# --- NEW COMPONENT: The Missing CNN ---
class QueryPreprocessor(keras.Model):
    """
    Processes raw observations (Images) to create the context vector Z'q
    which serves as the input to the Generated Target Network.
    """
    def __init__(self, output_dim):
        super().__init__()
        # Simple Convolutional Stack to replicate "CNN process's raw observations"
        self.cnn_stack = keras.Sequential([
            layers.Conv2D(16, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            # Projecting to the exact dimensionality expected by the Target Network
            layers.Dense(output_dim, activation='linear') 
        ])

    def call(self, x_raw):
        return self.cnn_stack(x_raw)

class QueryEncoder(keras.Model):
    """Generates Weights (gen_params) based on Cluster Context."""
    def __init__(self, output_param_count):
        super().__init__()
        self.intermediate_dim = 98 
        self.net = keras.Sequential([
            layers.Dense(self.intermediate_dim, activation='relu'),
            layers.Dense(output_param_count, activation='linear') 
        ])
    def call(self, centroid_context):
        return self.net(centroid_context)

class DynamicTargetNetwork(layers.Layer):
    """
    Executes an MLP defined purely by gen_params.
    INPUT CHANGE: Now explicitly expects Z'q instead of Z.
    """
    def call(self, z_prime_q_input, generated_params):
        current_x = z_prime_q_input
        
        # Infer input dim dynamically
        current_dim = tf.shape(current_x)[-1] 
        
        offset = 0
        
        # Process Hidden Layers
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
            
        # Process Output Layer (Must map back to EMBEDDING_DIM for Residual Add)
        out_dim = EMBEDDING_DIM 
        w_size_out = current_dim * out_dim
        b_size_out = out_dim
        
        w_flat_out = generated_params[:, offset : offset + w_size_out]
        b_val_out  = generated_params[:, offset + w_size_out : offset + w_size_out + b_size_out]
        
        w_matrix_out = tf.reshape(w_flat_out, [tf.shape(current_x)[0], current_dim, out_dim])
        
        # Produces Z'
        z_prime_delta = tf.einsum('bi,bij->bj', current_x, w_matrix_out) + b_val_out
        
        return z_prime_delta

class HyperRetrieverSystem(Model):
    """
    Orchestrates:
    1. Static Encoding (Z) for Centroid Lookup & Memory Add
    2. Dynamic Encoding (Z'q) via Preprocessing CNN
    3. Hypernetwork Generation
    4. Residual Creation
    """
    def __init__(self, enc_layer, output_dim):
        super().__init__()
        self.output_dim = output_dim
        
        # 1. Static Encoder (Produces Z)
        self.encoder = enc_layer
        
        # 2. NEW: Query Preprocessor (Produces Z'q from Raw Input)
        self.preprocessor = QueryPreprocessor(output_dim)
        
        # 3. Hypernetwork (Produces weights)
        self.qe = QueryEncoder(TOTAL_PARAMS_NEEDED) 
        
        # 4. Target Network (Consumes Z'q + weights -> Z')
        self.target_net = DynamicTargetNetwork()
        
    def call(self, inputs):
        # --- PATH A: STATIC RETRIEVAL PATH ---
        # Raw X -> Frozen AE -> Z
        z = self.encoder(inputs) 
        
        # Get Context from Z
        z_norm = tf.linalg.l2_normalize(z, axis=1)
        c_norm = tf.linalg.l2_normalize(CENTROID_VECS, axis=1)
        sims = tf.matmul(z_norm, c_norm, transpose_b=True)
        best_idx = tf.argmax(sims, axis=-1)
        ctx_vec = tf.gather(CENTROID_VECS, best_idx)
        
        # Generate Parameters
        gen_params = self.qe(ctx_vec)
        
        # --- PATH B: DYNAMIC QUERY PATH (THE FIX) ---
        # Raw X -> NEW CNN -> Z'q
        z_prime_q = self.preprocessor(inputs)
        
        # Run Target Network using Z'q as the ACTUAL input to the generated network
        # This produces Z' (the delta/correction)
        z_prime_delta = self.target_net(z_prime_q, gen_params)
        
        # --- FUSION ---
        # Q = Z (from Static) + Z' (from Dynamic)
        q = z + z_prime_delta
        
        # Final Memory Retrieval Logic
        q_norm = tf.linalg.l2_normalize(q, axis=1)
        mem_sim = tf.matmul(q_norm, MEM_BANK_VECS, transpose_b=True)
        
        values, indices = tf.math.top_k(mem_sim, k=NUM_NEIGHBORS)
        attn_weights = tf.nn.softmax(values, axis=1)
        
        neighbor_labels = tf.gather(MEM_BANK_LABELS, indices)
        pred_retrieval = tf.reduce_sum(
            tf.expand_dims(attn_weights, -1) * neighbor_labels, axis=1
        )
        return pred_retrieval

class FullGuidedSystem(Model):
    def __init__(self, hyper_ret, ve_path):
        super().__init__()
        self.retriever = hyper_ret
        print("Loading Value Encoder...")
        self.value_encoder = models.load_model(ve_path)
        self.value_encoder.trainable = False
        
    def call(self, inputs):
        p_ret = self.retriever(inputs)
        p_ve = self.value_encoder(inputs) # VE still takes raw input based on previous discussion
        return (p_ret * 0.8 + p_ve * 0.2)# / 2.0

# ---------------------------------------------------------
# INSTANTIATION & TRAINING
# ---------------------------------------------------------

loaded_obj = tf.saved_model.load(ENCODER_PATH)
frozen_enc_layer = FrozenEncoderLayer(loaded_obj)

hyper_system = HyperRetrieverSystem(frozen_enc_layer, EMBEDDING_DIM)
model = FullGuidedSystem(hyper_system, VALUE_ENC_PATH)

optimizer = Adam(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

print("\nStarting Training Loop with Corrected Architecture...")
dataset = tf.data.Dataset.from_tensor_slices((X_tr, y_tr_hot)).shuffle(5000).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    
    for step, (x_batch, y_true) in enumerate(dataset):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss_val = loss_fn(y_true, y_pred)
            
        grads = tape.gradient(loss_val, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        epoch_loss_avg.update_state(loss_val)
        epoch_acc_metric.update_state(y_true, y_pred)
        
        if step % 50 == 0:
             print(f"Step {step}: Loss={loss_val.numpy():.4f}, Acc={epoch_acc_metric.result().numpy():.4f}")

    print(f">> End Loss: {epoch_loss_avg.result().numpy():.4f}, Acc: {epoch_acc_metric.result().numpy():.4f}")

# ---------------------------------------------------------
# EVALUATION & SAVE
# ---------------------------------------------------------
print("\nEvaluating...")
pred_probs = model.predict(X_te)
pred_cls = np.argmax(pred_probs, axis=1)
acc = accuracy_score(y_te_int, pred_cls)
print(f"Final Accuracy: {acc:.4f}")

model.retriever.save(SAVE_PATH_HQE_SYSTEM)
print(f"Saved HQE System to: {SAVE_PATH_HQE_SYSTEM}")