from datetime import datetime, timedelta
import numpy as np
import os
import pandas as pd
import re
from sklearn.base import BaseEstimator


def create_folder(folder):
    "Utility function to create folder if it doesn't exist."
    if not os.path.exists(folder):
        os.makedirs(folder)
        

def make_dates(experiment_dates: dict) -> pd.DataFrame:
    """Create windows folds consisting of three sets {train, valid and test}.
    
    For example, the input will be a dictionary: 
        experiment_dates: 
            train_start: '2009-01-17'
            test_start: '2011-01-08'
            test_end: '2011-12-31'
            
    And return a dataframe:
    
    	train_start | train_end	 | valid_start | valid_end   | test_start | test_end
    ------------------------------------------------------------------------------
     	2009-01-17  | 2010-12-04 | 2010-12-11  | 2011-01-01	 | 2011-01-08 | 2011-01-29
    	2009-02-14  | 2011-01-01 | 2011-01-08  | 2011-01-29	 | 2011-02-05 | 2011-02-26
    
    Args
        experiment_dates: dictionary containing the train_start, test_start and test_end dates.
    Returns
        date_ranges: pd.DataFrame with the relevant dates for the experiment.
    
    """
    date_ranges = []
    i=0
    while True:        
        train_start = pd.to_datetime(experiment_dates['train_start']) + timedelta(weeks=4*i)
        train_end = pd.to_datetime(experiment_dates['test_start']) + timedelta(weeks=4*i) - timedelta(weeks=1) - timedelta(weeks=4)
        
        valid_start = pd.to_datetime(experiment_dates['test_start']) + timedelta(weeks=4*i)- timedelta(weeks=4)
        valid_end = pd.to_datetime(experiment_dates['test_start']) + timedelta(weeks=4*i) - timedelta(weeks=1)
        
        test_start = pd.to_datetime(experiment_dates['test_start']) + timedelta(weeks=4*i) 
        test_end = pd.to_datetime(experiment_dates['test_start']) + timedelta(weeks=(4*i)+4) - timedelta(weeks=1)

        dates_ = {'train_start': train_start,
                  'train_end': train_end,
                  'valid_start': valid_start,
                  'valid_end': valid_end,
                  'test_start': test_start,
                  'test_end':test_end}
        
        date_ranges.append(dates_)
        if test_end >= pd.to_datetime(experiment_dates['test_end']):
            break
        i+= 1
    return pd.DataFrame(date_ranges)


def make_lag_features(data: pd.DataFrame, window_size: int, 
                      col_name: str, prefix_name: str, inplace=True) -> pd.DataFrame:
    """
    Function that will generate the lag variables used in making predictions.
    
    Args
        data: pd.DataFrame, incoming dataset
        window_size: int, number of timesteps to look back t-window_size to t-1
        col_name: string, specify a column to look back on
        prefix_name: string, prefix of new column names
        inplace: bool
    Return
        data: pd.DataFrame
    """
    
    # If you don't want to change the data inplace, then you need to make a copy of the dataset
    if inplace: 
        dataset = data
    else:
        dataset = data.copy()
    
    for i in range(1, window_size + 1):
        dataset[f'{prefix_name}-{i}'] = dataset[col_name].shift(i)
    
    return dataset


def add_datepart(df, fldname, drop=True, time=False):
    "Function to convert transaction time-stamps into date features for XGBoost."
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    if drop: df.drop(fldname, axis=1, inplace=True)
    return df


def make_historical_avg(df: pd.DataFrame, r_list: list = [],
                        col_n: str = 'lag-1', google_trends: bool = False):
    """ Calculate multiple rolling historical mean using the lag-1 feature 
        to prevent leakage. This means there will be no need to drop 
        originital feature.
    
    Args
        df: pd.DataFrame of the original data
        r_list: list of rolling means to calculate
        col_n: the column name on which to make the rolling averages
        google_trends: boolean indicator to determine if the series have 
                       product categories or not
    Returns
        df: pd.DataFrame with added columns
    """
    # Iterate over specified rolling window sizes
    for r in r_list:
        # We reset the index to remember the initial index since the groupby will change the sequence 
        if google_trends:
            temp = df[col_n].rolling(r).mean().reset_index().rename(columns={'index':'level_1'})
        else:
            temp = df.groupby('product_category_name')[col_n].rolling(r).mean().reset_index()
        temp = temp.rename(columns={col_n:f'{col_n}-rolling-{r}'})
        # We use the column level_1 which was the initial index and set it back and sort it
        # This ensures we can simply pass in a column as a new column (no need to deal with joins)
        temp.set_index('level_1', inplace=True)
        temp.sort_index(inplace=True)
        # round up to two decimal points, since some of the averages may have many many decimals
        temp[f'{col_n}-rolling-{r}'] = temp[f'{col_n}-rolling-{r}'].apply(lambda x: np.round(x,2))
        # Create a new column with the respective rolling window column
        df[f'{col_n}-rolling-{r}'] = temp[f'{col_n}-rolling-{r}']
    return df


def fix_col_syntax(df: pd.DataFrame) -> pd.DataFrame:
    """ Used to clean the column names by either replacing unwanted 
        characters with underscores or deleting them.
    Args
        df: pd.DataFrame containing columns that have special characters
    Returns
        df: pd.DataFrame with the column names cleaned
    """
    col_names = df.columns
    col_names = [cname.replace("'", '_') for cname in col_names]
    col_names = [cname.replace(" ", '_') for cname in col_names]
    col_names = [cname.replace(":", '') for cname in col_names]
    col_names = [cname.replace("(", '') for cname in col_names]
    col_names = [cname.replace(")", '') for cname in col_names]
    col_names = [cname.replace("&", '') for cname in col_names]
    col_names = [cname.replace("-", '') for cname in col_names]
    df.columns = col_names
    return df


def parse_name(remove_items: list, fname: str) -> str:
    """ Used to clean the column names by removing substrings
        from the original string.
    Args
        remove_items: list of substrings
        fname: original string
    Returns
        clean fname without the substrings
    """
    for i in range(len(remove_items)):
        fname = fname.replace(remove_items[i], '')
    return fname