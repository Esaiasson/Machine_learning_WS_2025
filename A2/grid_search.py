import sys
import pandas as pd
import time
from sklearn.model_selection import KFold
import prediction_evaluation as eval
import itertools
#import regression_tree as rt



def grid_search_cv(df, target, model, pred_function, param_grid):
    
    #BORROWED FROM: https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    keys, values = zip(*param_grid.items())
    combinations_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
    kf = KFold(n_splits=2, shuffle=True)

    scores = []    
    results = []

    for i, combo in enumerate(combinations_list):
        print(f"Running combo: {i+1}")
        start = time.perf_counter()
        for train_index, test_index in kf.split(df):
            df_train_index = df.index[train_index]
            df_test_index = df.index[test_index]
            created_model = model(df.loc[df_train_index, ], target, combo, split_criterion=combo["split_criterion"])
            predictions = pred_function(created_model, df.loc[df_test_index, ])
            score = eval.rmse(df.loc[df_test_index, target], predictions)
            scores.append(score)
            
        end = time.perf_counter()
        mean_score = eval.cross_validation_score(scores)
        
        results.append({"runtime": end - start, "mean_score": mean_score, "Parameters": combo})
        
    results_df = pd.DataFrame.from_dict(results)
    sorted_results_df = results_df.sort_values("mean_score", ignore_index=True)
    best_model = model(df, target, sorted_results_df.loc[0,"Parameters"], split_criterion="sse")
    return best_model, results_df.loc[:5,]
   



#food_waste_df_train, food_waste_df_test = pre_food_wastage.preprocessing_food_wastage()
#grid_search_new(food_waste_df_train, "wastage_food_amount", rt.regression_tree, rt.predict_from_tree, param_grid)