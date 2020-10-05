import pandas as pd
import keras
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

'''
This method explod the noun phrases into rows against employee number and filter out
the skills using LSTM text classifier model. Vectorization has been
done by using GloVe vector. Refer LSTMtextClassifier repo for model training.
'''

def skill_filter(tokenizer_model, classifier_model, skill_df, prob, pad_len):
    all_skills = skill_df.explode('skill')
    all_skills_col = all_skills['skill'].astype('str')

    skill_vector = tokenizer_model.text_to_sequences(all_skills_col)

    skill_vector = pad_sequences(skill_vector, padding = 'post', maxlen = int(pad_len))

    skill_pred = classifier_model.predict(skill_vector)

    result = list(zip(all_skills['employee_number'], all_skills_col, skill_pred[:,0]))
    final_df = pd.DataFrame(result, columns = ['employee_number', 'skill', 'prob'])
    high_prob_df = final_df.loc[final_df['prob'] >= float(prob)]
    high_prob_df = high_prob_df.astype({'employee_number': str})

    return high_prob_df