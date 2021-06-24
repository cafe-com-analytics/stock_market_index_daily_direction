import numpy as np
import pandas as pd


def create_shifted_rt(df: pd.DataFrame, rts: list) -> pd.DataFrame:
    for t in rts:
        df[f"rt-{t}"] = df["rt"].shift(periods=t)
    return df


def uniform_clustering(df: pd.DataFrame, lst_columns: list) -> pd.DataFrame:
    """This function creates the target "Cluster" according to the limits described in article."""
    for column in lst_columns:
        conditions  = [
            df[column] < -1.12,
            (df[column] >= -1.12) & (df[column] < -0.42),
            (df[column] >= -0.42) & (df[column] < 0),
            (df[column] >= 0) & (df[column] < 0.44),
            (df[column] >= 0.44) & (df[column] < 1.07),
            df[column] >= 1.07]
        
        choices = [ 1, 2, 3, 4, 5, 6]
        df["cluster_"+column] = np.select(conditions, choices, default=np.nan)
    
    return df