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
    range1 = ["1995", "1994", "1993", "1992", "1991", "1990", "1989", "1988"]
    range2 = ["2007", "2006", "2005", "2004", "2003", "2002", "2001", "2000", "1999", "1998", "1997", "1996"]
    range3 = ["2020", "2019", "2018", "2017", "2016", "2015", "2014", "2013", "2012", "2011", "2010", "2009", "2008"]
    range4 = ["2023", "2022"]


    df["1984"] = df["1984"].fillna(14) #Only the 13 best schools where ranked, so the unranked once gets rank 14
    df["1986"] = df["1986"].fillna(11) #Only the 10 best schools where ranked, so the unranked once gets rank 11
    df[range1] = df[range1].fillna(26) #Only the 25 best schools where ranked, so the unranked once gets rank 26
    df[range2] = df[range2].fillna(51) #Only the 50 best schools where ranked, so the unranked once gets rank 51
    df[range3] = df[range3].fillna(151) #Only the 150 best schools where ranked, so the unranked once gets rank 151
    df["2021"] = df["2021"].fillna(51) #Only the 50 best schools where ranked, so the unranked once gets rank 51
    df[range4] = df[range4].fillna(151) #Only the 150 best schools where ranked, so the unranked once gets rank 151

    return df

def apply_processing(df, states_map):
    
    df = df.drop(["University Name", "IPEDS ID"], axis=1)

    df = pre.map_attribute(df, "State", states_map)

    df = non_rank_handling(df)
    df = pre.rename_columns(df)

    return df


def preprocessing_college_ranking():

    df = pd.read_csv("data/US-News-Rankings-Universities-Through-2023.csv")
    states_map = states_mapping(df)
    print("Dataset University Ranking")
    df_train, df_test = pre.train_test_split(df)
    pre.df_info(
        df, 
        df_train, 
        df_test, 
        "2023")
    

    df_train_processed = apply_processing(df_train, states_map)
    df_test_processed = apply_processing(df_test, states_map)

    
    return df_train_processed, df_test_processed
    







    

    

