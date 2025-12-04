import pandas as pd
import sys
import regression_tree as rt
import prediction_evaluation as eval
sys.path.append("preprocessing")
import preprocessing_food_wastage as pre_food_wastage



food_waste_df_train, food_waste_df_test = pre_food_wastage.preprocessing_food_wastage()


#TEMP FOR DEBUG
stop = [
    {
        "stopping_criterion": "max_depth",
        "stop_value": 2
    },
    {
        "stopping_criterion": "min_samples_split",
        "stop_value": 500
    }
]

print("TREE STARTED")
tree = rt.regression_tree(food_waste_df_train, "wastage_food_amount", stop, split_criterion="sse")
predictions = rt.predict_from_tree(tree, food_waste_df_test)
print("PREDICTIONS CREATED")
rmse = eval.rmse(food_waste_df_test["wastage_food_amount"], predictions)
print(rmse)

