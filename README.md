<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
<h1>
  Structuring Radiology Reports: Challenging LLMs with Lightweight Models
</h1>
</div>

<p align="center">
📝 <a href="https://arxiv.org/abs/2506.00200" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/collections/StanfordAIMI/structuring-with-lightweight-models-683e9eb895d42e04112fad88" target="_blank">Hugging Face</a> • 🧩 <a href="https://github.com/jomoll/rad-report-structuring" target="_blank">Github</a> • 🪄 <a href="https://stanford-aimi.github.io/structuring.html" target="_blank">Project</a>
</p>

<div align="center">
</div>

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
<img width="600" alt="image" src="https://github.com/user-attachments/assets/ff37c096-6376-4535-aceb-9c9d2d095235" />
</p>

### Models
| Model    | Variant      | HuggingFace Link |
|----------|--------------|-------------------|
| BERT2BERT | RoBERTa-base | [🤗 StanfordAIMI/SRR-BERT2BERT-RoBERTa-base](https://huggingface.co/StanfordAIMI/SRR-BERT2BERT-RoBERTa-base) |
|          | RoBERTa-biomed | [🤗 StanfordAIMI/SRR-BERT2BERT-RoBERTa-biomed](https://huggingface.co/StanfordAIMI/SRR-BERT2BERT-RoBERTa-biomed) |
|          | RoBERTa-PM-M3 | [🤗 StanfordAIMI/SRR-BERT2BERT-RoBERTa-PM-M3](https://huggingface.co/StanfordAIMI/SRR-BERT2BERT-RoBERTa-PM-M3) |
|          | RadBERT      | [🤗 StanfordAIMI/SRR-BERT2BERT-RadBERT](https://huggingface.co/StanfordAIMI/SRR-BERT2BERT-RadBERT) |
| T5       | T5-Base      | [🤗 StanfordAIMI/SRR-T5-Base](https://huggingface.co/StanfordAIMI/SRR-T5-Base) |
|          | Flan-T5      | [🤗 StanfordAIMI/SRR-T5-Flan](https://huggingface.co/StanfordAIMI/SRR-StanfordAIMI/SRR-T5-Flan) |
|          | SciFive      | [🤗 StanfordAIMI/SRR-T5-SciFive](https://huggingface.co/StanfordAIMI/SRR-T5-SciFive) |

---

### Dataset

| Dataset       | HuggingFace Link |
| -------------- | ---------------- |
| SRRG-Findings  | [🤗 StanfordAIMI/srrg_findings](https://huggingface.co/datasets/StanfordAIMI/srrg_findings) |

---

### Example Usage

```python
import io
import torch
from transformers import EncoderDecoderModel, AutoTokenizer

# Step 1: Setup
model_name = "StanfordAIMI/SRR-BERT2BERT-RoBERTa-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Load Processor and Model
model = EncoderDecoderModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right", use_fast=False)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.bos_token_id = tokenizer.cls_token_id
model.eval()

# Step 3: Inference (example from MIMIC-CXR dataset)
input_text = "CHEST RADIOGRAPH PERFORMED ON ___  ...  Impression: Limited exam with small bilateral effusions, cardiomegaly, and possible mild interstitial edema."
inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id)
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs["attention_mask"].to(device)

generated_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=286,
    min_new_tokens=120,
    decoder_start_token_id=model.config.decoder_start_token_id,
    num_beams=5,
    early_stopping=True,
    max_length=None
)[0]

decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(decoded)
```

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

## ✏️ Citation

```
@article{structuring-2025,
  title={Structuring Radiology Reports: Challenging LLMs with Lightweight Models},
  author={Moll, Johannes and Fay, Louisa and Azhar, Asfandyar and Ostmeier, Sophie and Lueth, Tim and Gatidis, Sergios and Langlotz, Curtis and Delbrouck, Jean-Benoit},
  journal={arXiv preprint arXiv:2506.00200},
  url={https://arxiv.org/abs/2506.00200},
  year={2025}
}

```
