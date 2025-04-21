## Structuring Radiology Reports: Challenging LLMs with Lightweight Models

### Project Overview
Radiology reports are critical for clinical decision-making but often lack a standardized format, limiting both human interpretability and machine learning (ML) applications. While large language models (LLMs) like GPT-4 can effectively reformat these reports, their proprietary nature, computational demands, and data privacy concerns limit clinical deployment. To address this challenge, we employed lightweight encoder-decoder models (<300M parameters), specifically T5 and BERT2BERT, to structure radiology reports from the MIMIC-CXR and CheXpert databases. We benchmarked our lightweight models against five open-source LLMs (3-8B parameters), which we adapted using in-context learning (ICL) and low-rank adaptation (LoRA) finetuning. We found that our best-performing lightweight model outperforms all ICL-adapted LLMs on a human-annotated test set across all metrics (BLEU: 212%, ROUGE-L: 63%, BERTScore: 59%, F1-RadGraph: 47%, GREEN: 27%, F1-SRRG-Bert: 43%). 
While the overall best-performing LLM (Mistral-7B with LoRA) achieved a marginal 0.3% improvement in GREEN Score over the lightweight model, this required $10\times$ more training and inference time, resulting in a significant increase in computational costs and carbon emissions. Our results highlight the advantages of lightweight models for sustainable and efficient deployment in resource-constrained clinical settings.

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
  <img src="https://github.com/user-attachments/assets/50c70f2a-fdc9-4130-9f40-18129b48e7a0" alt="domainadapt2" width="45%" style="margin-right: 10px;"/>
  <img src="https://github.com/user-attachments/assets/b513a01c-0edf-49a2-8fd9-9bc4f7a97994" alt="llmadaptation2" width="45%"/>

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
