import regression_tree as rt
import numpy as np
import pandas as pd
import math


def bootstrap(df, tree_id):
    '''
    generate the bootstrap sample from original data (with replacement) + oob_sample
    Parameters:
        df: dataset to build bootstrap sample from
        tree_id: reproducibility parameter, used to calculate the seed
    '''
    n = len(df)
    df_idx = list(df.index)

    np.random.seed(36 + tree_id)
    boot_idx = np.random.choice(df_idx, size=n, replace=True)

    # bootstrap data
    bootstrap_df = df.loc[boot_idx]

    # OOB = points not selected
    all_idx = set(df_idx)
    oob_idx = list(all_idx - set(boot_idx))
    oob_df = df.loc[oob_idx] if len(oob_idx) > 0 else np.array([])

    return bootstrap_df, oob_df


def populate_forest(df, target, stop, split_criterion, no_of_estimators, max_features_criterion):
    '''
    generate regression tree based on a bootstrap sample and polulates the forest
    Parameters:
        no_of_estimators: no of trees to be build
        max_features_criterion: criterion to select max split features
    '''
    # decide on the criterion to get max features to decide the split on
    total_features = len(df.columns)-1
    MAX_SPLIT_FEATURES = total_features
    if(max_features_criterion=='sqrt'):
        MAX_SPLIT_FEATURES = int(round(math.sqrt(total_features),0))
    elif(max_features_criterion=='log2'):
        MAX_SPLIT_FEATURES = int(round(math.log2(total_features), 0))

    forest = {}

    for tree_id in range(no_of_estimators):
        df_boot, df_oob = bootstrap(df, tree_id)

        # parallelize tree building
        tree = rt.regression_tree(
            df_boot,
            target,
            stop,
            split_criterion,
            max_features=MAX_SPLIT_FEATURES
        )

        forest[tree_id] = {'estimator':tree, 'oob_data': df_oob.index}

    return forest

def trees_mean(df):
    df['prediction'] = df.mean(axis=1, skipna=True)
    return df['prediction']

def eval_oob(forest, df, target):
    """
    stores the predictions from each tree of the forest
    """
    tree_preds = df[target].to_frame()
    
    for tree_id in forest.keys():     
        tree = forest[tree_id]['estimator']
        oob_data_idx = forest[tree_id]['oob_data']
        oob_data = df.loc[oob_data_idx]

        pred = rt.predict_from_tree(tree, oob_data)
        name = f'pred_{tree_id}'
        pred_series = pd.Series(pred, index = oob_data_idx, name=name).to_frame()
    
        tree_preds = tree_preds.join(pred_series, how='outer')
        del tree_preds[tree_preds.columns[0]]
        forest_preds = trees_mean(tree_preds)

    return forest_preds


def predict_from_forest(df, forest):
    '''
    predict on unseen data
    '''
    tree_preds = pd.DataFrame()
    tree_preds.index = df.index
    
    for tree_id in forest.keys():     
        tree = forest[tree_id]['estimator']

        pred = rt.predict_from_tree(tree, df)
        name = f'pred_{tree_id}'
    
        tree_preds[name] = pred
        forest_test_preds = trees_mean(tree_preds)
    
    return forest_test_preds
