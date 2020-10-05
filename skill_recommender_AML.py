import pandas as pd
import keras
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from skill_splitter import skill_splitter
from skill_matcher import skill_matcher
from skill_filter import skill_filter
import os
import azureml.core
from azureml.core import Workspace, Datastore
from azureml.data.data_reference import DataReference
from azureml.core import Dataset, Run
from azureml.core.model import Model

run = Run.get_context()
ws = run.experiment.workspace

def init():

    global skill_tokenizer
    global skill_classifier

    # Use models use in LSTMtextClassifier repo
    skill_tokenizer_model = Model.get_model_path('skill_tokenizer.pkl')
    skill_classifier_model = Model.get_model_path('skill_classifier')

    # Open tokenizer model
    with open(skill_tokenizer_model, 'rb') as handle:
        skill_tokenizer = pickle.load(handle)


def run(mini_batch):

    # Define parameters
    pad_len = 20
    skill_filter_prob = 0.8
    master_match_prob = 80

    skill_final_df = pd.DataFrame(columns=['employee_number', 'skill', 'match_skill', 'match_skill_prob'])

    # Call text splitter
    skill_df = skill_splitter(mini_batch)

    # Call skill filter
    skill_filter_df = skill_filter(
        skill_tokenizer,
        skill_classifier,
        skill_df,
        skill_filter_prob,
        pad_len
    )

    # Find closest matching skill in master skill set based on threshold
    ds_master = 'skill_master' # AML master skill set registered dataset
    df = Dataset.get_by_name(workspace=ws, name=ds_master) # Read master skill set registered in AML Datastore
    masterskill_df = df.to_pandas_dataframe()
    master_list = masterskill_df['NAME'].to_list() # Convert master skill to list

    skill_match_df = skill_matcher(skill_filter_df, master_list, master_match_prob)

    return skill_match_df


