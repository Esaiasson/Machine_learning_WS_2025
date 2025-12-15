import pandas as pd
import math
import random
from Node import Node
import numpy as np


def mean(series):
    '''
    Calculates the mean of a series
    Parameters:
        series: A pandas series
    '''
    n = series.size # Number of rows in the series
    return series.sum()/n


def sse(split_1, split_2):
    '''
    Calculates the SSE of two series
    Parameters:
        split_1: A pandas series
        split_2: A pandas series
    '''
    
    mean_split_1 = split_1.mean()
    mean_split_2 = split_2.mean() 

    sse_split_1 = ((split_1 - mean_split_1)**2).sum()
    sse_split_2 = ((split_2 - mean_split_2)**2).sum()
    
    sse = sse_split_1 + sse_split_2

    return sse



def mse(split_1, split_2):
    '''
    MSE splitting criterion
    Parameters:
        split_1: A pandas series
        split_2: A pandas series
    '''
    mean_split_1 = split_1.mean()
    mean_split_2 = split_2.mean() 


    mse_split_1 = (((split_1 - mean_split_1)**2).sum())/len(split_1)
    mse_split_2 = (((split_2 - mean_split_2)**2).sum())/len(split_2)
    
    mse = mse_split_1 + mse_split_2

    return mse



def splitting_measure(df, target, split_criterion, max_features=None):
    '''
    Finds the values for different splitting measures per attribute in the dataframe
    Parameters:
        df: A pandas DataFrame 
        target: A name of a column in the df that acts as the target attribute 
        split_criterion: name of method that will be used to evaluate the split
            Allowed values: "sse"
        max_features: Number of attributes to be considered for best split. If no value is supplied, the value will be equal to the number of attributes in the dataframe
    Returns:
        A dictionary of the best spliting criterion containing the following:
            "attribute": The attribute to split
            "split_value": The value to split the attribute by
            "below_or_equal_predict": The mean of the target attribute in the split below or equal to the split value
            "over_predict": The mean of the target attribute in the split over the split value
    '''
    
    attribute_split_impurity = {} #Dictionary for keeping track of the best split for each attribute 
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
            split_scores = {} #Dictionary for storing the sse for each split value
            for i in range(0,n): #Itterate over all split values
                if split_criterion == "sse":
                    #Calculates the SSE for each possible split, the sse is saved with the key of x for which the split is val > x
                    score = sse(split_candidate_sorted.loc[:i, target], split_candidate_sorted.loc[i: , target])
                elif split_criterion == "mse":
                    score = mse(split_candidate_sorted.loc[:i, target], split_candidate_sorted.loc[i: , target])

                #Store the results of the evaluation function as a value belonging the key which is the split value
                split_scores[split_candidate_sorted.loc[i,col]] = score

            min_score_split_value = min(split_scores, key=split_scores.get) #Gets the key having the minimum value
            min_score = split_scores.get(min_score_split_value)
            #Stores the SSE as the key, with the split value and attribute name
            attribute_split_impurity[min_score] = {"attribute": col, "split_value": min_score_split_value}

    best_split_sse =  min(attribute_split_impurity) #The minimum SSE 
    best_split_dict = attribute_split_impurity.get(best_split_sse) #Gets the nested dictionary corresponding to the min SSE value
    best_split_dict["mean"] = subset_mean # adds the mean of the y-values in the whole subset, which will be used for prediction
    
    return best_split_dict

def splitting_measure_delta(df, target, split_criterion, max_features=None):
    '''
    Finds the values for different splitting measures per attribute in the dataframe
    Parameters:
        df: A pandas DataFrame 
        target: A name of a column in the df that acts as the target attribute 
        split_criterion: name of method that will be used to evaluate the split
            Allowed values: "sse"
        max_features: Number of attributes to be considered for best split. If no value is supplied, the value will be equal to the number of attributes in the dataframe
    Returns:
        A dictionary of the best spliting criterion containing the following:
            "attribute": The attribute to split
            "split_value": The value to split the attribute by
            "below_or_equal_predict": The mean of the target attribute in the split below or equal to the split value
            "over_predict": The mean of the target attribute in the split over the split value
    '''
    
    attribute_split_impurity = {} #Dictionary for keeping track of the best split for each attribute 
    subset_mean = mean(df[target]) # Mean of target attribute in the dataframe
    
    predictor_attributes = df.loc[:, df.columns != target] #Dataframe without target attribute
    if max_features == None: 
        max_features = len(predictor_attributes.columns.tolist()) #If no max_features value is supplied, set it to the number of attributes in the dataframe

    selected_columns = random.sample(predictor_attributes.columns.tolist(), max_features) #Randomly select a subset of the dataframe, with the size of max_features

    for col in selected_columns:
        split_candidate = df[[col, target]] #Create a new dataframe with only one column and the target attribute
        split_candidate_sorted = split_candidate.sort_values(col,ignore_index=True) #Sort the dataframe by the predictor attribute
        y = split_candidate_sorted[target].values
        x = split_candidate_sorted[col].values
        n = len(y) #number of rows 

        count_left = 0
        sum_left = 0
        sum_sq_left = 0
        
        count_right = n
        sum_right = y.sum()
        sum_sq_right = np.dot(y, y)

        # initiailze the best_score and split tracker
        # best_score = float("inf")
        # best_split_value = None

        split_scores = {} #Dictionary for storing the sse for each split value
        for i in range(0,n): #Itterate over all split values
            value = y[i]

            count_left += 1
            sum_left += value
            sum_sq_left += value*value
            
            count_right -= 1
            sum_right -= value
            sum_sq_right -= value*value

            if count_right==0 or count_left==0:
                continue
            
            if split_criterion == 'sse':
                measure_on_left = sum_sq_left - ((sum_left*sum_left)/count_left)
                measure_on_right = sum_sq_right - ((sum_right*sum_right)/count_right)
            
            elif split_criterion == 'mse':
                measure_on_left = (sum_sq_left - (sum_left*sum_left)/count_left)/count_left
                measure_on_right = (sum_sq_right - (sum_right*sum_right)/count_right)/count_right
                        
            score = measure_on_left + measure_on_right

            #Store the results of the evaluation function as a value belonging the key which is the split value
            split_scores[split_candidate_sorted.loc[i,col]] = score

        min_score_split_value = min(split_scores, key=split_scores.get) #Gets the key having the minimum value
        min_score = split_scores.get(min_score_split_value)
        #Stores the SSE as the key, with the split value and attribute name
        attribute_split_impurity[min_score] = {"attribute": col, "split_value": min_score_split_value}

    best_split_sse =  min(attribute_split_impurity) #The minimum SSE 
    best_split_dict = attribute_split_impurity.get(best_split_sse) #Gets the nested dictionary corresponding to the min SSE value
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
        if "max_depth" in stop:
            if(current_depth <= stop["max_depth"]):
                pass
            else:
                return False
        if "min_samples_split" in stop:
            if(df_length >= stop["min_samples_split"]):
                pass
            else:
                return False
            
    return True


def build_tree(df, target, depth, stop, split_criterion, max_features):
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
        split_criterion: name of method that will be used to evaluate the split
            Allowed values: "sse"
        max_features: Number of attributes to be considered for best split. If no value is supplied, the value will be equal to the number of attributes in the dataframe
    Returns:
        An instance of the class Node
    '''
    if (evaluate_stopping_criterion(stop, df_length=len(df), current_depth=depth) == True): 
        best_split_dict = splitting_measure_delta(df, target, split_criterion, max_features) #Finds the optimal split
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
        node.set_left(build_tree(left_df, target, (depth + 1), stop, split_criterion, max_features)) 
        node.set_right(build_tree(right_df, target, (depth + 1), stop, split_criterion, max_features))

        return node
    else:
        return None
    

def regression_tree(df, target, stop, split_criterion, max_features=None):
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
        split_criterion: name of method that will be used to evaluate the split
            Allowed values: "sse"
        max_features: Number of attributes to be considered for best split. If no value is supplied, the value will be equal to the number of attributes in the dataframe
    '''

    tree = build_tree(df, target, 0, stop, split_criterion, max_features)
    return tree


def find_leaf_node(node, row):
    if row[node.attribute] <= node.split_value:
        if node.left != None:
            return find_leaf_node(node.left, row)
        elif node.left == None:
            return node.mean
    if row[node.attribute] > node.split_value:
        if node.right != None:
            return find_leaf_node(node.right, row)
        elif node.right == None:
            return node.mean



def predict_from_tree(tree, df):
    predicitons = []
    for row in df.index:
        pred = find_leaf_node(tree, df.loc[row])
        predicitons.append(pred)
    
    return predicitons
        
        






        