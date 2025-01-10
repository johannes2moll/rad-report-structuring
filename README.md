## Setup

### Installation Steps

Follow these steps to set up the environment and get the project running:

```bash
# Step 1: Clone the Repository
git clone https://github.com/johannes2moll/SRREval
cd SRREval

# Step 2: Create Conda Environment
conda create -n srreval python=3.10
conda activate srreval

# Step 3: Install Requirements
pip install -e src/StructEval
pip install -r requirements.txt

# Step 4: Prepare the Data and set HOME directory
# Set DIR in constants.py

# Step 5: Train a Model
bash train_model.sh

# Step 7: Generate Prediction on Test Set
bash run_model.sh

# Step 8: Evaluate
bash calc_metrics.sh
