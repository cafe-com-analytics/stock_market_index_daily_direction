import numpy as np
import pandas as pd
import yfinance as yf


def downloading_stocks_data(dct, start_date: str = "2011-01-01", end_date: str = "2022-01-01") -> pd.DataFrame:
    """
    Download the stocks daily information from tickers listed as keys of a dictionary, gets only "Close" price from
    each day within start_date and end_date.

    Args:
        dct (dict): format {'ticker': {'name': name, etc}}
        start_date (str, optional): [description]. Defaults to "2011-01-01".
        end_date (str, optional): [description]. Defaults to "2022-01-01".

    Returns:
        pd.DataFrame: dataframe of close prices of each ticker.
    """
    df = yf.download(list(dct.keys())[0], start=start_date, end=end_date, show_errors=False)[["Close"]]
    df.columns = [dct[list(dct.keys())[0]]["name"]]

    for market_index in list(dct.keys())[1:]:
        df_temp = yf.download(market_index, start=start_date, end=end_date)[["Close"]]
        df_temp.columns = [dct[market_index]["name"]]
        df = df.merge(df_temp, how='left', left_index=True, right_index=True)

    df.dropna(how='all', axis=0, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    return df


def daily_return(df, lst_columns: list = 'all') -> pd.DataFrame:
    """
    Return the daily return of the lst_columns.
    """
    if lst_columns == 'all':
        df.columns = df.columns.tolit()
    elif isinstance(lst_columns, list):
        pass
    else:
        lst_columns = list(lst_columns)

    for column in lst_columns:
        df[column] = (np.log(df[column]) - np.log(df[column].shift(periods=1)))*100

    return df


def return_in_period(df, lst_columns: list = 'all') -> pd.DataFrame:
    """
    Return the return of the lst_columns.
    """
    if lst_columns == 'all':
        df.columns = df.columns.tolit()
    elif isinstance(lst_columns, list):
        pass
    else:
        lst_columns = list(lst_columns)

    for column in lst_columns:
        df[column] = df[column]/df[column][0]

    return df


def create_shifted_rt(df: pd.DataFrame, rts: list) -> pd.DataFrame:
    for t in rts:
        df[f"rt-{t}"] = df["rt"].shift(periods=t)
    return df


def uniform_clustering(df: pd.DataFrame, lst_columns: list) -> pd.DataFrame:
    """This function creates the target "Cluster" according to the limits described in article."""
    for column in lst_columns:
        conditions = [
            df[column] < -1.12,
            (df[column] >= -1.12) & (df[column] < -0.42),
            (df[column] >= -0.42) & (df[column] < 0),
            (df[column] >= 0) & (df[column] < 0.44),
            (df[column] >= 0.44) & (df[column] < 1.07),
            df[column] >= 1.07]

        choices = [1, 2, 3, 4, 5, 6]
        df["cluster_"+column] = np.select(conditions, choices, default=np.nan)

    return df
