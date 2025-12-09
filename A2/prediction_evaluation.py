import pandas as pd
import math

def rmse(target, predictions):
    '''
    Calculates the RMSE score of a a target attribute and corresponding predictions
    Parameters: 
        target: The target attribute, can be a pandas series, dataframe or an array 
        target: The predictions, can be a pandas series, dataframe or an array 

    '''

    if len(target) != len(predictions):
        raise Exception("Length of target and predictions does not match") 
    if isinstance(target, (pd.Series, pd.DataFrame)) == False:
        target = pd.Series(target)
    if isinstance(predictions, (pd.Series, pd.DataFrame)) == False:
        predictions = pd.Series(predictions, index=target.index)
    
    total_error = (sum((target - predictions)**2))
    rmse_score = math.sqrt((total_error/(len(target))))

    return rmse_score


def cross_validation_score(scores):
    mean_score = sum(scores)/len(scores)
    return mean_score
