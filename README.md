## Setup

### Installation Steps

Follow these steps to set up the environment and get the project running:

```bash
# Step 1: Clone the Repository
git clone https://github.com/johannes2moll/rad-report-structuring.git
# Optional: If submodule doesn't work (StructEval folder doesn't exist in src), clone submodule
cd rad-report-structuring/src
git clone https://github.com/jbdel/StructEval.git

# Step 2: Create Conda Environments 
# To reproduce all results, three different environments are needed (due to version collisions of green_score, radgraph, and transformers.EncoderDecoder)
# srrrun: training and running models: run_llm.sh, run_model.sh, train_llm.sh, train_model.sh
# srreval: evaluate all metrics but GREEN: calc_metrics.sh
# green: evaluate on GREEN metric: calc_metrics.sh (Note that for this you have to activate the import in src/StructEval/structueval/StructEval.py) and change the parameters in src/calc_metrics.py
conda create -n srrrun python=3.10
conda create -n srreval python=3.10
conda create -n green python=3.10.0

# Step 3: Install Requirements
conda activate srrrun
pip install -r requirements_run.txt
conda activate srreval 
pip install -e src/StructEval
pip install -r requirements_eval.txt
conda activate green
pip install -r requirements_green.txt

# Step 4: Prepare the Data and set HOME directory
# Set DIR and DIR_MODELS_TUNED in src/constants.py

# Step 5: Train a Model
conda activate srrrun
bash train_model.sh
bash train_llm.sh

# Step 7: Generate Prediction on Test Set
conda activate srrrun
bash run_model.sh
bash run_llm.sh

# Step 8: Evaluate
conda activate srreval
bash calc_metrics.sh
