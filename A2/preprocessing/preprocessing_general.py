import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def df_info(full_df, train_df, test_df, target):
    """
    Prints dataframe info and plots distribution histograms for full, train, and test datasets
    in a single row of subplots.

    Parameters:
        full_df: Full dataset (pandas DataFrame)
        train_df: Training split (pandas DataFrame)
        test_df: Testing split (pandas DataFrame)
        target: Name of the target column (str)
    """

    # Print basic info
    for split_name, df in zip(["Full", "Train", "Test"], [full_df, train_df, test_df]):
        print(f"{split_name} dataframe: {df.shape[0]} rows, {df.shape[1]} attributes")
        if split_name == "Full":
            na_rows = df.isnull().any(axis=1).sum()
            print(f"Rows containing NA values before preprocessing: {na_rows}")

    # Create subplot figure with 1 row and 3 columns
    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True,
        subplot_titles=["Full dataset", "Train dataset", "Test dataset"]
    )

    datasets = [full_df, train_df, test_df]
    colors = ["#636EFA", "#EF553B", "#00CC96"]

    for i, df in enumerate(datasets, start=1):
        fig.add_trace(
            go.Histogram(
                x=df[target],
                nbinsx=10,
                name=f"{['Full','Train','Test'][i-1]}",
                marker_color=colors[i-1]
            ),
            row=1,
            col=i
        )

    # Update layout
    fig.update_layout(
        title_text=f"Distribution of target attribute: {target} in Full, Train, and Test splits",
        height=500,
        width=1200,
        showlegend=False
    )
    
    # Update axes labels
    for i in range(1,4):
        fig.update_xaxes(title_text=target, row=1, col=i)
        fig.update_yaxes(title_text="Frequency", row=1, col=i)

    fig.show()
    


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