import pandas as pd
import regression_tree as rt
import prediction_evaluation as eval


food_waste = pd.read_csv("data/food_wastage_data.csv")

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

tree = rt.regression_tree(food_waste[["Quantity of Food", "Number of Guests", "Wastage Food Amount"]], "Wastage Food Amount", stop, split_criterion="sse")
predictions = rt.predict_from_tree(tree, food_waste[["Quantity of Food", "Number of Guests"]])

print(predictions)