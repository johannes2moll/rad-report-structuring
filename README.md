## Structuring Radiology Reports: Challenging LLMs with Lightweight Models

### Project Overview
Radiology reports are critical for clinical
decision-making but often lack a standard-
ized format, limiting both human interpretabil-
ity and machine learning (ML) applications.
While large language models (LLMs) have
shown strong capabilities in reformatting clini-
cal text, their high computational requirements,
lack of transparency, and data privacy con-
cerns hinder practical deployment. To ad-
dress these challenges, we explore lightweight
encoder-decoder models (<300M parame-
ters)—specifically T5 and BERT2BERT—for
structuring radiology reports from the MIMIC-
CXR and CheXpert Plus datasets. We bench-
mark these models against eight open-source
LLMs (1B–70B parameters), adapted using
prefix prompting, in-context learning (ICL),
and low-rank adaptation (LoRA) finetuning.
Our best-performing lightweight model out-
performs all LLMs adapted using prompt-
based techniques on a human-annotated test set.
While some LoRA-finetuned LLMs achieve
modest gains over the lightweight model on
the Findings section (BLEU 6.4%, ROUGE-L
4.8%, BERTScore 3.6%, F1-RadGraph 1.1%,
GREEN 3.6%, and F1-SRR-BERT 4.3%),
these improvements come at the cost of sub-
stantially greater computational resources. For
example, LLaMA-3-70B incurred more than
400 times the inference time, cost, and car-
bon emissions compared to the lightweight
model. These results underscore the poten-
tial of lightweight, task-specific models as sus-
tainable and privacy-preserving solutions for
structuring clinical text in resource-constrained
healthcare settings.

### Task
Automatically transform free-text chest X-ray radiology reports into a standardized, structured format.
<p align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/c988cc9b-12f3-4fcb-93d6-a4737dbf7f27" />
</p>

### Models
<p align="center">
<img src="https://github.com/user-attachments/assets/65222bdb-7e44-4c21-a95a-56ccde323223" alt="models" width="800"/>
</p>
<h3>Results</h3>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ff7dfea-034f-4aa5-9ee8-12cb12af6dd2" alt="domainadapt2" width="45%" style="margin-right: 10px;"/>

  <img src="https://github.com/user-attachments/assets/a5abe771-70fc-4194-bb33-75ba06235174" alt="llmadaptation2" width="45%"/>


<img src="https://github.com/user-attachments/assets/98066f08-7712-473a-9c0b-bb95457f32ee" alt="qualitative4" width="600"/>
</p>


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
```

## Citation
```
@misc{moll2025structuring,
  title        = {Structuring Radiology Reports: Challenging LLMs with Lightweight Models},
  author       = {Moll, Johannes and Fay, Louisa and Azhar, Asfandyar and Ostmeier, Sophie and Lueth, Tim and Gatidis, Sergios and Langlotz, Curtis P, Delbrouck, Jean-Benoit},
  year         = {2025},
  note         = {Under review},
}
```
