import pandas as pd
from textblob import TextBlob
import re
import nltk
import subprocess
import sys
import os

'''
This method splits text into noun phrases and return
the extracted noun phrases against employee_number
'''

# Download textblob corpus into the run environment
os.system(f"python -m textblob.download_corpora")

# Pass dataframe with column names as employee_number and skill_text
def skill_splitter(df_skill):
    df_skill = df_skill.astype(str)
    notremovelist = "#+"
    df_skill['skill_text'] = df_skill['skill_text'].map(lambda x: re.sub(r'[^\w'+notremovelist+']', ' ', x))

    employee_number = []
    skill =[]

    for _, row in df_skill.iterrow():
        employee_number.append(row.employee_number)
        skill.append(TextBlob(str(row.skill_text)).noun_phrases)

    split_list = pd.DataFrame({
        "employee_number": employee_number,
        "skill": skill
    })

    return split_list