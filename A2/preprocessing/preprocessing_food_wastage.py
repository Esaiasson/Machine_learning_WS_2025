import pandas as pd
import random
import preprocessing_general as pre

def apply_processing(df):
    
    columns_for_ohe = ['Type of Food', 'Event Type', 'Storage Conditions', 'Seasonality', 'Preparation Method', 'Geographical Location']
    df = pre.one_hot_encode(df,columns_for_ohe)
    
    mapping_purchase = {
        'Occasional': 1,
        'Regular': 2,
    }
    
    mapping_pricing = {
        'Low': 1,
        'Moderate': 2,
        'High': 3
    }
    
    df = pre.map_attribute(df, "Purchase History", mapping_purchase)
    df = pre.map_attribute(df, "Pricing", mapping_pricing)
    df = pre.rename_columns(df)

    return df


def preprocessing_food_wastage():
    
    df = pd.read_csv("data/food_wastage_data.csv")
    print("Dataset Food Wastage")
    
    df_train, df_test = pre.train_test_split(df)
    
    pre.df_info(
        df, 
        df_train, 
        df_test, 
        "Wastage Food Amount")
    
    df_train_processed = apply_processing(df_train)
    df_test_processed = apply_processing(df_test)
    
    return df_train_processed, df_test_processed