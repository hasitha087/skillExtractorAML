from fuzzywuzzy import fuzz, process
import pandas as pd

'''
This method matches the extracted skill with master skill set (defined skill set)
and return the closest matching skill based on the probability
threshold using Levenstine Distance based algorithm.
'''

def skill_matcher(df, master_list, prob):
    employee_number = []
    skill = []
    match_skill = []
    match_skill_prob = []

    for _, row in df.iterrow():
        highest_prob = process.extractOne(str(row.skill), master_list, scorer=fuzz.token_set_ratio)
        if(highest_prob[1] > float(prob)):
            employee_number.append(row.employee_number)
            skill.append(row.skill)
            match_skill.append(highest_prob[0])
            match_skill_prob.append(highest_prob[1])

    skill_matcher_df = pd.DataFrame({
        "employee_numner": employee_number,
        "skill": skill,
        "match_skill": match_skill,
        "match_skill_prob": match_skill_prob
    })

    return skill_matcher_df