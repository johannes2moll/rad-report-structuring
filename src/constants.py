import os

##############################################################
### directories ##############################################

# TODO: set DIR_PROJECT as location for all data and models
DIR_PROJECT = "/your/project/directory/here/"
assert os.path.exists(DIR_PROJECT), "please enter valid directory"

# TODO: for best practice, move data to DIR_PROJECT/data/ (outside repo)
DIR_DATA = os.path.join(DIR_PROJECT, "data/")  # input data
if not os.path.exists(DIR_DATA):
    DIR_DATA = "data/"

# TODO: set directory of tuned models. created automatically
DIR_MODELS_TUNED =   # tuned models
