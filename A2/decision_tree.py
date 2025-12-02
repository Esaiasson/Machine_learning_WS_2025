import pandas as pd
import math


food_waste = pd.read_csv("data/food_wastage_data.csv")


def mean(series):
    n = series.size # Number of rows in the series
    return series.sum()/n

def sse(split_1, split_2):
    
    mean_split_1 = mean(split_1)
    mean_split_2 = mean(split_2) 

    print(mean_split_1)    
    print(mean_split_2)


def splitting_measure(df, target):
    '''
    Finds the values for different splitting measures per attribute in the dataframe
    Parameters:
        df: A pandas DataFrame 
    '''
    attribute_variance = {}
    for col in df.columns:
        split_candidate = df[[col, target]]
        split_candidate_sorted = split_candidate.sort_values(col,ignore_index=True)
        n = split_candidate_sorted.size
        split_variance = {}
        for i in range(0,n):
            sse(split_candidate_sorted.loc[:i, target], split_candidate_sorted.loc[(i): , target])
            #if attribute_sorted.loc[(i+1):]: #Ensures the split actually consists of values
                #print(attribute_sorted.loc[(i+1):])
            #attribute_variance[col] = std_dev(attribute_sorted)
    #print(attribute_variance)


def std_dev(attribute):
    '''
    Calculates the standard deviation of a attribute
    Parameters: 
        Attribute: A pandas Series
    Returns:
        standard_devation: A value for the standard deviation of the series
    '''
    numeric_attribute = pd.to_numeric(attribute) # Converts the attribute to numeric
    n = numeric_attribute.size # Number of rows in the series
    mean = numeric_attribute.sum()/n # Calculates the mean of the series
    mean_deviation = 0 # variable for storing the total deviation from the mean
    for value in numeric_attribute: #Calculates the deviation from the mean for every value in the series
        mean_deviation = mean_deviation + ((value-mean)**2) 
    standard_deviation = math.sqrt(mean_deviation/(n-1)) # Formula for standard deviation
    
    return standard_deviation



    

def decision_tree(df, target):
    '''
    Creates a decision tree
    Parameters: 
        df: A pandas DataFrame
    '''
    df_temp = df.apply(pd.to_numeric) #TEMPORARY FOR DEBUGGING
    splitting_measure(df_temp, target)
        

decision_tree(food_waste[["Quantity of Food", "Number of Guests", "Wastage Food Amount"]], "Wastage Food Amount")





        