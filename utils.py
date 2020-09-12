import pandas as pd
import numpy as np
import re

def make_lag_features(data, window_size, col_name, prefix_name, inplace=True):
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
    
    for i in range(1,window_size + 1):
        dataset[f'{prefix_name}-{i}'] = dataset[col_name].shift(i)
    
    return dataset


#Function to convert transaction time-stamps into date features for XGBoost 
#https://github.com/fastai/fastai_old/blob/master/dev_nb/x_009a_rossman_data_clean.ipynb
#https://docs.fast.ai/tabular.transform.html#add_datepart
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


    