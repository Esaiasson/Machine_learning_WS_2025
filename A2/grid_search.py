import pandas as pd
import time
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
import prediction_evaluation as eval
import random_forest as rf
import itertools

def grid_combinations(param_grid):
    #BORROWED FROM: https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    keys, values = zip(*param_grid.items())
    combinations_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations_list


def evaluate_combo(combo, df, target, model, pred_function, kf):
    start = time.perf_counter()
    rmse_scores = []
    mae_scores = []    

    for train_index, test_index in kf.split(df):
        df_train_index = df.index[train_index]
        df_test_index = df.index[test_index]
        created_model = model(df.loc[df_train_index, ], target, combo, split_criterion=combo["split_criterion"])
        predictions = pred_function(created_model, df.loc[df_test_index, ])
        rmse, mae = eval.measure_predictions(df.loc[df_test_index, target], predictions)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
    end = time.perf_counter()
    mean_rmse = eval.cross_validation_score(rmse_scores)
    mean_mae = eval.cross_validation_score(mae_scores)
    

    return {
        "runtime": end - start,
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
        "Parameters": combo
    }
    
def grid_search_cv2(df, target, model, pred_function, param_grid):
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

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(evaluate_combo)(combo, df, target, model, pred_function, kf)
        for combo in combinations_list
    )

    results_df = pd.DataFrame(results)
    sorted_results_df = results_df.sort_values("mean_rmse", ignore_index=True, ascending=True)
    best_model = model(df, target, sorted_results_df.loc[0,"Parameters"], split_criterion=sorted_results_df.loc[0,"Parameters"]["split_criterion"])
    return best_model, sorted_results_df





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

    results = []

    for i, combo in enumerate(combinations_list):
        print(f"Running combo: {i+1} out of: {len(combinations_list)}")
        start = time.perf_counter()
        rmse_scores = []
        mae_scores = []    

        for train_index, test_index in kf.split(df):
            df_train_index = df.index[train_index]
            df_test_index = df.index[test_index]
            created_model = model(df.loc[df_train_index, ], target, combo, split_criterion=combo["split_criterion"])
            predictions = pred_function(created_model, df.loc[df_test_index, ])
            rmse, mae = eval.measure_predictions(df.loc[df_test_index, target], predictions)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            
        end = time.perf_counter()
        mean_rmse = eval.cross_validation_score(rmse_scores)
        mean_mae = eval.cross_validation_score(mae_scores)
        
        results.append({"runtime": end - start, "mean_rmse": mean_rmse, "mean_mae": mean_mae,"Parameters": combo})
        
    results_df = pd.DataFrame.from_dict(results)
    sorted_results_df = results_df.sort_values("mean_rmse", ignore_index=True, ascending=True)
    best_model = model(df, target, sorted_results_df.loc[0,"Parameters"], split_criterion=sorted_results_df.loc[0,"Parameters"]["split_criterion"])
    return best_model, sorted_results_df




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
      scoring={
        "rmse": make_scorer(eval.rmse, greater_is_better=False),
        "mae": make_scorer(eval.mae, greater_is_better=False)
      },
      refit="rmse",
      verbose=True
  )

  grid_search.fit(x,y)

  results = pd.DataFrame(grid_search.cv_results_)
  results["mean_rmse"] = abs(results["mean_test_rmse"])
  results["mean_mae"] = abs(results["mean_test_mae"])
  results_orderd = results.sort_values("mean_rmse", ignore_index=True, ascending=True)
  elapsed = time.perf_counter() - start
  results_orderd["runtime"] = elapsed
  print(grid_search.best_estimator_)
  print("Time(s): ", elapsed)
  results_orderd.rename(columns={'params':'Parameters'}, inplace=True)
  return results_orderd.loc[:,["runtime", "mean_rmse", "mean_mae", "Parameters"]], grid_search.best_estimator_
   


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

def grid_search_oob_scikit(df, target_attribute, param_grid):
    
    x = df.loc[:, df.columns != target_attribute]
    y = df[target_attribute]
    results = []
    forests = []

    combinations_list = grid_combinations(param_grid)

    for i, combo in enumerate(combinations_list):
        print(f"Running combo: {i+1} out of: {len(combinations_list)}")
        start = time.perf_counter()
        forest = RandomForestRegressor(
            random_state=1,
            **combo,
            bootstrap=True,
            oob_score=True,
            # n_jobs=-1,
        )
        forest.fit(x,y)

        oob_preds = forest.oob_prediction_

        oob_rmse = eval.rmse(y, oob_preds)
        oob_mse = eval.mse(y, oob_preds)
        end = time.perf_counter()

        forests.append(forest)
        results.append({"runtime": end - start, "mean_rmse": oob_rmse, "mean_mse": oob_mse, "Parameters": combo, "index": i})

    results_df = pd.DataFrame.from_dict(results)
    sorted_results_df = results_df.sort_values("mean_rmse", ignore_index=True, ascending=True)
    best_model = forests[sorted_results_df.loc[0, "index"]]
    return sorted_results_df.drop(labels="index", axis=1), best_model




def knn(df, target_attribute):
    x = df.loc[:, df.columns != target_attribute]
    y = df[target_attribute]
    start = time.perf_counter()

    knn = KNeighborsClassifier()
    knn.fit(x, y)
    end = time.perf_counter()
    runtime = (end - start)
    
    return knn, runtime 
        
