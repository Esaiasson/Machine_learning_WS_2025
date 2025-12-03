import pandas as pd
import math
import random
from Node import Node


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
    




def splitting_measure(df, target, max_features=None):
    '''
    Finds the values for different splitting measures per attribute in the dataframe
    Parameters:
        df: A pandas DataFrame 
        target: A name of a column in the df that acts as the target attribute 
        max_features: Number of attributes to be considered for best split. If no value is supplied, the value will be equal to the number of attributes in the dataframe
    Returns:
        A dictionary of the best spliting criterion containing the following:
            "attribute": The attribute to split
            "split_value": The value to split the attribute by
            "below_or_equal_predict": The mean of the target attribute in the split below or equal to the split value
            "over_predict": The mean of the target attribute in the split over the split value
    '''
    
    attribute_sse = {} #Dictionary for keeping track of the best split for each attribute 
    subset_mean = mean(df[target]) # Mean of target attribute in the dataframe
    
    predictor_attributes = df.loc[:, df.columns != target] #Dataframe without target attribute
    if max_features == None: 
        max_features = len(predictor_attributes.columns.tolist()) #If no max_features value is supplied, set it to the number of attributes in the dataframe

    selected_columns = random.sample(predictor_attributes.columns.tolist(), max_features) #Randomly select a subset of the dataframe, with the size of max_features

    for col in selected_columns:
        if col != target: #Don't calculate SSE for the target column
            split_candidate = df[[col, target]] #Create a new dataframe with only one column and the target attribute
            split_candidate_sorted = split_candidate.sort_values(col,ignore_index=True) #Sort the dataframe by the predictor attribute
            n = len(split_candidate_sorted) #number of rows 
            split_sse = {} #Dictionary for storing the sse for each split value
            for i in range(0,n): #Itterate over all splot values
                #Calculates the SSE for each possible split, the sse is saved with the key of x for which the split is val > x
                sse_value = sse(split_candidate_sorted.loc[:i, target], split_candidate_sorted.loc[i: , target])
                #Store the results of the SSE function as a value belonging the key which is the split value
                split_sse[split_candidate_sorted.loc[i,col]] = sse_value

            min_sse_split_value = min(split_sse, key=split_sse.get) #The gets the key having the minimum value
            min_sse = split_sse.get(min_sse_split_value)
            #Stores the SSE as the key, with the split value and attribute name
            attribute_sse[min_sse] = {"attribute": col, "split_value": min_sse_split_value}

    best_split_sse =  min(attribute_sse) #The minimum SSE 
    best_split_dict = attribute_sse.get(best_split_sse) #Gets the nested dictionary corresponding to the min SSE value
    best_split_dict["mean"] = subset_mean # adds the mean of the y-values in the whole subset, which will be used for prediction
    
    return best_split_dict



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

def evaluate_stopping_criterion(stop, df_length, current_depth=None):
    '''
    Evaluates a given stopping criterion. Will always return False if df_length is < 1
    Params:
        stop: A list of dictionaries containing the stoping criterion.
        df_length: Length of the dataframe
        current_depth: Depth of the tree
            Default: None
    Returns:
        True or False
    '''

    if (df_length < 1): #Always stop if the DataFrame contains less then 1 row
        return False
    else:
        for stop_dict in stop:
            if stop_dict["stopping_criterion"] == "max_depth":
                if(current_depth <= stop_dict["stop_value"]):
                    pass
                else:
                    return False
            elif stop_dict["stopping_criterion"] == "min_samples_split":
                if(df_length >= stop_dict["stop_value"]):
                    pass
                else:
                    return False
                
        return True
        


def build_tree(df, target, depth, stop):
    '''
    Creates linked instances of the class Node
    Parameters:
        df: A pandas DataFrame
        target: A column name in df that will be the class to be predicted
        depth: The current depth of the node 
        stop: A list of dictionaries containing the stoping criterion.
        Allowed keys and values in each dictionary:
            stopping_criterion: A string describing the stopping criterion 
                Allowed values: "max_depth", "min_samples_split"
            stop_value: The value for which the stopping criteria will be evaulated against
    Returns:
        An instance of the class Node
    '''
    if (evaluate_stopping_criterion(stop, df_length=len(df), current_depth=depth) == True): #Run if stoping criterion is unfullfilled
        best_split_dict = splitting_measure(df, target) #Finds the optimal split
        print(f"Split by: {best_split_dict}, depth: {depth} ") #TEMP FOR DEBUG
        #Creates an instance of the class node
        node = Node(
            best_split_dict.get("attribute"),
            best_split_dict.get("split_value"),
            best_split_dict.get("mean"),
            depth
        )
        
        left_df = df[df[node.attribute] <= node.split_value] #Values of the dataframe up to and including the split value
        right_df = df[df[node.attribute] > node.split_value] #Values of the dataframe over the split value

        #Recursivly creates left and right nodes 
        node.set_left(build_tree(left_df, target, (depth + 1), stop)) 
        node.set_right(build_tree(right_df, target, (depth + 1), stop))

        return node
   
    

def regression_tree(df, target, stop):
    '''
    Creates a decision tree
    Parameters: 
        df: A pandas DataFrame
        target: A column name in df that will be the class to be predicted
        stop: A list of dictionaries containing the stoping criterion.
        Allowed values:
            stopping_criterion: A string describing the stopping criterion 
                Allowed values: "max_depth", "min_samples_split"
            stop_value: The value for which the stopping criteria will be evaulated against
    '''


    df_temp = df.apply(pd.to_numeric) #TEMPORARY FOR DEBUGGING

    tree = build_tree(df_temp, target, 0, stop)
    print(tree)        



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

regression_tree(food_waste[["Quantity of Food", "Number of Guests", "Wastage Food Amount"]], "Wastage Food Amount", stop)





        