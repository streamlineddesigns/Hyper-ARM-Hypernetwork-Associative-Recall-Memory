#create virtual enviornment
python3 -m venv tf_env

#install dependencies
pip install --upgrade pip
pip install chromadb==0.4.22 posthog==2.4.0 pysqlite3-binary
pip install tensorflow pandas numpy scikit-learn matplotlib

#activate python venv
source tf_env/bin/activate

#[K Branch] train ae
python train_ae_cnn.py

#[V Branch] train ve mlp predictions
python train_ve_mlp_via_ae.py

#[H Memory] populate vd via ae
python vd_populate_via_ae.py
python kmeans.py

#inspect vd
python inspect_vd.py

#[Q Branch] run hyper qe trainer
python hqe_trainer.py

#validation
python hyper_system_validation.py