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