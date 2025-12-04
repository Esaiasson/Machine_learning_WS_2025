import pandas as pd
import math

def rmse(target, predictions):
    '''
    Calculates the RMSE score of a a target attribute and corresponding predictions
    Parameters: 
        target: The target attribute, can be a pandas series, dataframe or an array 
        target: The predictions, can be a pandas series, dataframe or an array 

    '''
    if isinstance(target, (pd.Series, pd.DataFrame)) == False:
        target = pd.Series(target)
    if isinstance(predictions, (pd.Series, pd.DataFrame)) == False:
        predictions = pd.Series(predictions)

    eval_df = pd.concat([target, predictions], axis=1)
    eval_df.columns = ["target", "prediction"]
    total_error = (sum((eval_df["target"] - eval_df["prediction"])**2))
    rmse_score = math.sqrt((total_error/(len(eval_df))))

    return rmse_score
