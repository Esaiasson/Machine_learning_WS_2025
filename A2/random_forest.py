import regression_tree
import numpy as np
import pandas as pd
import math

def get_random_max_features(df):
    total_features = df.size
    max_random_features_split_criterion = math.sqrt(total_features)

def bootstrap(df):
    '''
    generate the bootstrap sample from original data (with replacement) + oob_sample
    '''
    n = len(df)
    
    boot_idx = np.random.choice(np.arange(n), size=n, replace=True)

    # bootstrap data
    bootstrap_df = df[boot_idx]

    # OOB = points not selected
    all_idx = set(range(n))
    oob_idx = list(all_idx - set(boot_idx))
    oob_df = df[oob_idx] if len(oob_idx) > 0 else np.array([])

    return bootstrap_df, oob_df


def populate_forest(no_of_estimators, df, max_features_criterion, target, stop, split_criterion):
    '''
    generate regression tree based on a bootstrap sample and polulates the forest
    Parameters:
        no_of_estimators: no of trees to be build
        max_features_criterion: criterion to select max split features
    '''

    # decide on the criterion to get max features to decide the split on
    total_features = df.size
    MAX_SPLIT_FEATURES = df.size
    if(max_features_criterion=='sqrt'):
        MAX_SPLIT_FEATURES = math.sqrt(total_features)
    elif(max_features_criterion=='log2'):
        MAX_SPLIT_FEATURES = math.log2(total_features)

    for i in no_of_estimators:
        bootstrap(df)
        regression_tree.regression_tree(df, target, 0, stop, split_criterion, max_features=MAX_SPLIT_FEATURES)
        i += 1



def predict_from_forest():
    '''
    implement majority voting on forest trees to predict
    '''

    

def eval_oob(oob_data):
    '''
    keep track of out of bag samples and compute oob_score
    '''
