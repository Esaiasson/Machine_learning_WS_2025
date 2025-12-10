import pandas as pd
import re
import matplotlib.pyplot as plt


def df_info(df, target, split):
    print(f"The {split} dataframe consists of {df.shape[0]} rows, and {df.shape[1]} attributes")
    if split == "full":
        na_rows = df.isnull().any(axis=1).sum()
        print(f"Rows containing NA values before preprocessing: {na_rows}")    

    plt.figure(figsize=(8, 6))
    plt.hist(df[target], bins=10)
    plt.title(f'Distribution of target attribute in {split} dataset: {target}')
    plt.tight_layout()
    plt.show()
    


def one_hot_encode(df, columns_for_ohe):
    return pd.get_dummies(df, columns=columns_for_ohe).replace({True: 1, False: 0})


def map_attribute(df, col, mapping):

    df[col] = df[col].map(mapping)
    return df

def rename_columns(df):
    df = df.rename(columns={element: re.sub(r'\s',r'_', element) for element in df.columns.tolist()})
    df = df.rename(str.lower, axis='columns')
    return df


def train_test_split(df):
    
    train_df = df.sample(frac=2/3, random_state=42)
    test_df = df.drop(train_df.index)
    
    return train_df, test_df