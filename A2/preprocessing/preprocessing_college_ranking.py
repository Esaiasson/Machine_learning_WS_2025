import pandas as pd
import random
import preprocessing_general as pre
import os 

def states_mapping(df):
    states_map = {}
    states = pd.unique(df["State"])
    for i, state in enumerate(states):
        states_map[state] = i
    
    return states_map


def non_rank_handling(df):
    df["1984"] = df["1984"].fillna(14) #Only the 13 best schools where ranked, so the unranked once gets rank 14
    df["1986"] = df["1986"].fillna(11) #Only the 10 best schools where ranked, so the unranked once gets rank 11
    df
    return df

def apply_processing(df, states_map):
    
    df = df.drop(["University Name", "IPEDS ID"], axis=1)

    df = pre.map_attribute(df, "State", states_map)

    df = non_rank_handling(df)
    
    return df


def preprocessing_college_ranking():

    df = pd.read_csv("../data/US-News-Rankings-Universities-Through-2023.csv")
    states_map = states_mapping(df)
    df_train, df_test = pre.train_test_split(df)
    
    df_train_processed = apply_processing(df_train, states_map)
    df_test_processed = apply_processing(df_test, states_map)
    pre.df_info(df_train_processed)

    
    return df_train_processed, df_test_processed
    

preprocessing_college_ranking()





    

    

