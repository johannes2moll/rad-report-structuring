import os

### SET DIRECTORY PATHS ###
# TODO: set DIR_PROJECT as location for all data and models
DIR_PROJECT = '.'#"/your/project/directory/here/"
assert os.path.exists(DIR_PROJECT), "please enter valid directory"

# TODO: for best practice, move data to DIR_PROJECT/data/ (outside repo)
DIR_DATA = os.path.join(DIR_PROJECT, "data/")  # input data
if not os.path.exists(DIR_DATA):
    DIR_DATA = "data/"

# TODO: set directory of tuned models. created automatically
DIR_MODELS_TUNED = "/path/where/models/are/saved/"  # tuned models

# directory of physionet's pre-trained models (clin-t5, clin-t5-sci) and facebook's RoBERTa-base-PM-M3-Voc-distill-align
# download here: https://www.physionet.org/content/clinical-t5/1.0.0/
# download here: https://github.com/facebookresearch/bio-lm/blob/main/README.md
DIR_MODELS_CLIN = os.path.join(DIR_PROJECT, "models/")

### MODELS ###
MODELS = {
    "roberta-base": "FacebookAI/roberta-base",
    "roberta-biomed": "allenai/biomed_roberta_base",
    "roberta-PM": os.path.join(DIR_MODELS_CLIN, "RoBERTa-base-PM-M3-Voc-distill-align"),
    "roberta-rad": "UCSD-VA-health/RadBERT-RoBERTa-4m",

    "t5-base": "t5-base",
    "flan-t5-base": "google/flan-t5-base",
    "scifive-base": "razent/SciFive-base-Pubmed_PMC",
    "clin-t5-sci": os.path.join(DIR_MODELS_CLIN, "Clinical-T5-Sci"),
    "clin-t5-base": os.path.join(DIR_MODELS_CLIN, "Clinical-T5-Base"),
}

LLMS = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "medalpaca-7b": "medalpaca/medalpaca-7b",
    "medalpaca-13b": "medalpaca/medalpaca-13b",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3": "microsoft/Phi-3.5-mini-instruct",
}