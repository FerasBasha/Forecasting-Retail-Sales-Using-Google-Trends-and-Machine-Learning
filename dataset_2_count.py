import os
import pandas as pd
from utils import metrics, utils
import numpy as np


def _filter_date_state(df: pd.DataFrame, state, start_date, end_date, exclude_delivered=True):
    """
    Function that will parse the data and return the desired format.
    
    Args
        df: pd.dataframe, sales_order_with_payments
        state: string, refering to the customer_state
        state_date: start date to filter
        end_date: end date to filter
        exclude_delivered: determines if it excludes transactions that were not delivered
    Returns
        sp_sales: pd.Dataframe, daily sales over time
    """
    
    # Filter state, dates and number of maximum number of payement installments
    # (df['customer_state'] == state) & isin(top10_states) %(df['customer_state'] == state) &
    
    # states should be a string, else you need to use .isin(states_list)
    df_sample = df[(df['customer_state'] == state) &
                  (df['order_approved_at'] > start_date) &
                  (df['order_approved_at'] <= end_date) &
                  (df['payment_installments'] < 2 )]
    
    if exclude_delivered: 
        df_sample = df_sample[df_sample['order_status'] == 'delivered']
  
    return df_sample

def _make_historical_avg(df, r_list=[]):
    """Calculate multiple rolling historical mean using the lag-1 feature to prevent leakage.
    
    Args: 
        df: pd.DataFrame
        h_list: list of historical mean to calculate
    Returns:
        df
    """
    # Iterate over specified rolling window sizes
    for r in r_list:
        # We reset the index to remember the initial index since the groupby will change the sequence 
        temp = df.groupby('product_category_name')['lag-1'].rolling(r).mean().reset_index()
        temp = temp.rename(columns={'lag-1':f'rolling-{r}'})
        # We use the column level_1 which was the initial index and set it back and sort it
        # This ensures we can simply pass in a column as a new column (no need to deal with joins)
        temp.set_index('level_1',inplace=True)
        temp.sort_index(inplace=True)
        # round up to two decimal points, since some of the averages may have many many decimals
        temp[f'rolling-{r}'] = temp[f'rolling-{r}'].apply(lambda x: np.round(x,2))
        # Create a new column with the respective rolling window column
        df[f'rolling-{r}'] = temp[f'rolling-{r}']
    return df

def make_dataset(configs):
    """
    Function that will create the dataset based on specified configurations.
    
    Args
        configs: dict, dictionary containing all necessary configurations
    Returns
        tuple of training, validation and testing sets
    """
    
    # assign configs related to input files and read data
    configs_tables = configs['tables']
    customers = pd.read_csv(os.path.join(configs['directories']['base_dir'], 
                                         configs_tables['customers']))
    products = pd.read_csv(os.path.join(configs['directories']['base_dir'], 
                                        configs_tables['products']))
    pc_name_trans = pd.read_csv(os.path.join(configs['directories']['base_dir'], 
                                                           configs_tables['pc_name_trans']))
    orders = pd.read_csv(os.path.join(configs['directories']['base_dir'], 
                                      configs_tables['orders']))
    order_items = pd.read_csv(os.path.join(configs['directories']['base_dir'], 
                                           configs_tables['orderitems']))
    order_payments = pd.read_csv(os.path.join(configs['directories']['base_dir'], 
                                              configs_tables['orderpayments']))

    # store product categories' translations in a dictionary 
    # and translate the product category column in the products table to english
    pc_name_trans = pc_name_trans.set_index('product_category_name')['product_category_name_english'].to_dict()
    products['product_category_name'] = products['product_category_name'].map(pc_name_trans)

    # join tables together
    sales_order = pd.merge(orders, customers, on='customer_id', how='inner')
    sales_order_item = order_items.merge(sales_order, on = 'order_id', how = 'left')                     
    sales_order_full = sales_order_item.merge(products,on = 'product_id',how = 'inner')
    sales_order_with_payments = sales_order_full.merge(order_payments, on='order_id')
    
    # convert date column to datatime object
    sales_order_with_payments['order_approved_at'] = pd.to_datetime(sales_order_with_payments['order_approved_at'], 
                                                                    format='%Y-%m-%d') 
    
    # filter dataframe
    df = _filter_date_state(sales_order_with_payments, 
                            configs['state'], 
                            configs['start_date'], 
                            configs['end_date'])

    # group by product categories and aggregate by number of daily transactions
    df = (df.set_index('order_approved_at')
             .groupby(['product_category_name'])
             .resample('d')["payment_value"]
             .count()
             .reset_index())

    # assumption that in retail, days without data are due to no sales
    df.fillna(0, inplace=True)

    # add historical payment value features by product categories
    df = (df.groupby('product_category_name')
             .apply(utils.make_lag_features,
                    window_size=configs['pre_processing']['window_size'],
                    col_name='payment_value',
                    prefix_name='lag')
             .reset_index(drop=True))
    
    # add date-specific features using fastai's function
    if configs['pre_processing']['add_date_features']:
        df = utils.add_datepart(df, fldname='order_approved_at', drop=False)

    # filter out product categories
    if len(configs['product_categories'])>0:
        df = df[df['product_category_name'].isin(configs['product_categories'])]
    if configs['pre_processing']['rolling_history']:
        df = _make_historical_avg(df,configs['pre_processing']['r_list'])
        
        
    # ff we encode, then we would also need to return the mapping for the encoding
    if configs['rm_product_category']:
        _ = df.pop('product_category_name')
    
    # split dataset into train, valid, test
#     train_df = df[(df['order_approved_at'] >= pd.Timestamp(configs['dates']['train_start'])) &
#                     (df['order_approved_at'] < pd.Timestamp(configs['dates']['valid_start']))].copy()
    
    if configs['dates']['valid_start']:
        train_df = df[(df['order_approved_at'] >= pd.Timestamp(configs['dates']['train_start'])) &
                    (df['order_approved_at'] < pd.Timestamp(configs['dates']['valid_start']))].copy()
        valid_df = df[(df['order_approved_at'] >= pd.Timestamp(configs['dates']['valid_start'])) &
                    (df['order_approved_at'] < pd.Timestamp(configs['dates']['test_start']))].copy()
    else:
        train_df = df[(df['order_approved_at'] >= pd.Timestamp(configs['dates']['train_start'])) &
                      (df['order_approved_at'] < pd.Timestamp(configs['dates']['test_start']))].copy()
    test_df = df[(df['order_approved_at'] >= pd.Timestamp(configs['dates']['test_start'])) &
                   (df['order_approved_at'] < pd.Timestamp(configs['dates']['test_end']))].copy()
    
    train_df.sort_values('order_approved_at',inplace = True)
    if configs['dates']['valid_start']: valid_df.sort_values('order_approved_at',inplace=True)
    test_df.sort_values('order_approved_at',inplace = True)
    
    # drop columns that are no longer required
    if configs['drop_date']:
        train_df.drop(columns=['order_approved_at'], inplace = True)
        if configs['dates']['valid_start']:
            valid_df.drop(columns=['order_approved_at'], inplace = True)
        test_df.drop(columns=['order_approved_at'], inplace = True)
    
    # if only interested in the daily sales, then filter all other columns
    # usefull for time series methods are require univariate inputs
    if configs['univariate']: 
        train_df = train_df['payment_value']
        if configs['dates']['valid_start']:
            valid_df = valid_df['payment_value']
        test_df = test_df['payment_value']
    if configs['dates']['valid_start']:
        return (train_df, valid_df, test_df)
    else:
        return (train_df,test_df)