#create virtual enviornment
python3 -m venv tf_env

#activate python venv
source tf_env/bin/activate

#install dependencies
pip install --upgrade pip
pip install chromadb==0.4.22 posthog==2.4.0 pysqlite3-binary
pip install tensorflow pandas numpy scikit-learn matplotlib

#EQV Attention

#[E Branch] train ae
python train_ae_cnn.py

#[V Branch] train ve mlp predictions
python train_ve_mlp_via_ae.py

#[Q Branch] trains hyper qe trainer
./run_continuous_training_loop.sh