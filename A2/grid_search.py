import pandas as pd
import time
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
import prediction_evaluation as eval
import random_forest as rf
import itertools

def grid_combinations(param_grid):
    #BORROWED FROM: https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    keys, values = zip(*param_grid.items())
    combinations_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations_list

def grid_search_cv(df, target, model, pred_function, param_grid):
    '''
    Function to perform grid search of a set of hyperparameters with cross validation
    Parameters:
        df: Dataframe to train and evalute on 
        target: Attribute in the dataframe that is the target
        model: A regression model
        pred_function: The method to make predictions
        param_grid: A dictionary of parameters to test
    '''
    
    combinations_list = grid_combinations(param_grid)
            
    kf = KFold(n_splits=5, shuffle=True)

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
    sorted_results_df = results_df.sort_values("mean_score", ignore_index=True, ascending=True)
    best_model = model(df, target, sorted_results_df.loc[0,"Parameters"], split_criterion=sorted_results_df.loc[0,"Parameters"]["split_criterion"])
    return best_model, sorted_results_df.loc[:5,]




def grid_search_scikit(df, target_attribute, param_grid):

  x = df.loc[:, df.columns != target_attribute]
  y = df[target_attribute]

  start = time.perf_counter()

  tree = DecisionTreeRegressor(random_state=1)

  cv_strategy = KFold(
    n_splits=5,
    shuffle=True,
    random_state=1
  )

  grid_search = GridSearchCV(
      estimator=tree,
      param_grid=param_grid,
      cv=cv_strategy,
      scoring=make_scorer(eval.rmse, greater_is_better=False),
      refit=True,
      verbose=True
  )

  grid_search.fit(x,y)

  results = pd.DataFrame(grid_search.cv_results_)
  results["mean_test_score"] = abs(results["mean_test_score"])
  results_orderd = results.sort_values("mean_test_score", ignore_index=True, ascending=True)
  elapsed = time.perf_counter() - start
  results_orderd["runtime"] = elapsed
  print(grid_search.best_estimator_)
  print("Time(s): ", elapsed)
  return results_orderd.loc[:5,["runtime", "mean_test_score", "params"]], grid_search.best_estimator_
   


def grid_search_obb(df, target_attribute, param_grid):
    
    results = []
    forests = []

    combinations_list = grid_combinations(param_grid)

    for i, combo in enumerate(combinations_list):
        print(f"Running combo: {i+1} out of: {len(combinations_list)}")
        start = time.perf_counter()
        forest = rf.populate_forest(
            df,
            target_attribute,
            combo,
            split_criterion=combo["split_criterion"],
            no_of_estimators=combo["no_of_estimators"],
            max_features_criterion=combo["max_features_criterion"]
        )
        predictions = rf.eval_oob(forest, df, target_attribute)
        pred_and_target = df[target_attribute].to_frame().join(predictions.to_frame(), how='outer')
        pred_and_target = pred_and_target.dropna()
        score = eval.rmse(pred_and_target[target_attribute], pred_and_target["prediction"])
        end = time.perf_counter()

        forests.append(forest)
        results.append({"runtime": end - start, "mean_score": score, "Parameters": combo, "index": i})

    results_df = pd.DataFrame.from_dict(results)
    sorted_results_df = results_df.sort_values("mean_score", ignore_index=True, ascending=True)
    best_model = forests[sorted_results_df.loc[0, "index"]]
    return best_model, sorted_results_df.loc[:5,].drop(labels="index", axis=1)

    
        
