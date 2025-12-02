import pandas as pd
import math


food_waste = pd.read_csv("data/food_wastage_data.csv")


def mean(series):

    n = series.size # Number of rows in the series
    return series.sum()/n

def sse(split_1, split_2):
    
    mean_split_1 = mean(split_1)
    mean_split_2 = mean(split_2) 

    sse_split_1 = 0
    sse_split_2 = 0
    for y in split_1: 
        sse_split_1 += (y-mean_split_1)**2
    for y in split_2: 
        sse_split_1 += (y-mean_split_2)**2
    
    sse = (sse_split_1 + sse_split_2)

    return sse    
    




def splitting_measure(df, target):
    '''
    Finds the values for different splitting measures per attribute in the dataframe
    Parameters:
        df: A pandas DataFrame 
    '''
    attribute_sse = {}
    for col in df.columns:
        if col != target:
            split_candidate = df[[col, target]]
            split_candidate_sorted = split_candidate.sort_values(col,ignore_index=True)
            n = len(split_candidate_sorted)
            split_sse = {}
            
            for i in range(0,n):
                #Calculates the sse for each possible split, the sse is saved with the key of x for which the split is val > x
                split_sse[split_candidate_sorted.loc[i,col]] = sse(split_candidate_sorted.loc[:i, target], split_candidate_sorted.loc[i: , target])
            
            min_sse_split_value = min(split_sse, key=split_sse.get)
            attribute_sse[split_sse.get(min_sse_split_value)] = {"attribute": col, "split_value": min_sse_split_value}
    
    best_split_sse =  min(attribute_sse)
    best_split = attribute_sse.get(best_split_sse)
    split_by_attribute = best_split["attribute"]
    split_by_value = best_split["split_value"]
    return split_by_attribute, split_by_value



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



    

def regression_tree(df, target):
    '''
    Creates a decision tree
    Parameters: 
        df: A pandas DataFrame
    '''
    df_temp = df.apply(pd.to_numeric) #TEMPORARY FOR DEBUGGING
    split_by_attribute, split_by_value = splitting_measure(df_temp, target)
    print(f"Split by: {split_by_attribute}, at: {split_by_value}")
        

regression_tree(food_waste[["Quantity of Food", "Number of Guests", "Wastage Food Amount"]], "Wastage Food Amount")





        