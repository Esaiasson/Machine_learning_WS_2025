import pandas as pd
import re


def df_info(df):
    print(df.dtypes)
    

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

def drop_cols(df, indices=None):
    if indices is None:
        indices = []
    cols = df.columns[indices]
    df.drop(columns=cols, inplace=True)

    return df