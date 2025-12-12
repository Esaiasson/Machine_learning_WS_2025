import pandas as pd
import math


def rmse(actual, predicted):
    '''
    Calculates the RMSE score of a a target attribute and corresponding predictions
    Parameters: 
        target: The target attribute, can be a pandas series, dataframe or an array 
        target: The predictions, can be a pandas series, dataframe or an array 
    '''
    n = len(actual)
    total_error = sum((actual - predicted)**2)
    rmse_score = math.sqrt((total_error/n))

    return rmse_score

def mse(actual, predicted):
    '''
    Calculates the MSE score of a a target attribute and corresponding predictions
    Parameters: 
        target: The target attribute, can be a pandas series, dataframe or an array 
        target: The predictions, can be a pandas series, dataframe or an array 
    '''
    n = len(actual)
    mse_score = (sum((actual - predicted)**2))/n
    return mse_score

def mae(actual, predicted):
    '''
    Calculates the MAE score of a a target attribute and corresponding predictions
    Parameters: 
        target: The target attribute, can be a pandas series, dataframe or an array 
        target: The predictions, can be a pandas series, dataframe or an array 
    '''

    n = len(actual)
    mae_score = (sum(abs(actual - predicted)))/n
    return mae_score


def measure_predictions(target, predictions):
    '''
    Calculates a given performance measure on predictions
    Parameters: 
        target: The target attribute, can be a pandas series, dataframe or an array 
        target: The predictions, can be a pandas series, dataframe or an array 
        measure: The measure to calculate on the predictions
    '''

    if len(target) != len(predictions):
        raise Exception("Length of target and predictions does not match") 
    if isinstance(target, (pd.Series, pd.DataFrame)) == False:
        target = pd.Series(target)
    if isinstance(predictions, (pd.Series, pd.DataFrame)) == False:
        predictions = pd.Series(predictions, index=target.index)

    rmse_score = rmse(target, predictions)
    mae_score = mae(target, predictions)
    
    return rmse_score, mae_score


def cross_validation_score(scores):
    mean_score = sum(scores)/len(scores)
    return mean_score



def big_table_formating(dataset, all_results, rmse_own, mae_own, rmse_scikit, mae_scikit):
    
    all_results.loc[
        (all_results["dataset"] == dataset) & (all_results["implementation"] == "Ours"),
        ["test set rmse", "test set mae"]
    ] = rmse_own, mae_own
    all_results.loc[
        (all_results["dataset"] == dataset) & (all_results["implementation"] == "Scikit"),
        ["test set rmse", "test set mae"]
    ] = rmse_scikit, mae_scikit
    return all_results
