# Can Google Trends help make better forecasts? 📈

| ![image](https://user-images.githubusercontent.com/39706513/101570472-74af5700-39a4-11eb-8811-c0ed824542bd.png) | 
|:--:| 
| *Example of XGBoost predictions for Pretzels (Breakfast At The Frat)* |


### Project Structure

```
├── README.md          <- The top-level README for developers using this project.
│
├── conf               <- Configurations folder for the project.
│   ├── catalog.yml    <- Contains paths to reference datasets.
│   └── params.yml     <- Contains parameters for the experiment and models.
│
├── data
│   ├── 01_raw         <- The original, immutable data dump.
│   ├── 02_processed   <- The final, canonical data sets for modeling.
│   ├── 03_external    <- Google Trends data.
│   ├── 04_results     <- Saved predictions for each model.
│   └── 05_extra       <- Additional outputs.
│
│
├── notebooks          <- Jupyter notebooks.
│   ├── breakfast      <- Contains the notebooks relevant for Breakfast at The Frat.
│   └── olist          <- Contains the notebooks relevant for Olist.
│
├── mlruns             <- Experiment Tracking logs.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── environment.yml    <- The conda environment file for reproducing the analysis environment.
│
├── src                <- Additional code for this project.
│   ├── utils.py       <- Main functions for use in this project.
│   ├── metrics.py     <- Metrics for use in this project.
│   ├── scalers.py     <- Custom classes to scale the data.
│   └── sarima.py      <- A Scikit-Learn SARIMA model wrapper for easier modelling.
```

To create the environment, do:

```
conda env create -f environment.yml
```

To activate the environment, do:

```
conda activate <env name>
```

### Configuration files

##### catalog.yml

Here are a few examples of paths that are used to reference the datasets. There are two major branches, olist and breakfast.

```yaml
olist:
    base_dir: "data/01_raw"
    tables:
        customers:     "olist_customers_dataset.csv"
        products:
    ...
    ...

breakfast:
    base_dir: "data/01_raw"
    xlsx_fname: "dunnhumby_breakfast.xlsx"
    sheet_names: 
        transactions: "dh Transaction Data"
    ...
    ...
```

##### params.yml

The experiment dates defined in this configuration file.   

```yaml
olist:
    experiment_dates: 
        train_start: '2017-01-01'
        test_start: '2018-01-07'
    ...
    ...

breakfast:
    dataset:
        store_ids: 
            2277: 'OH'       
            2277: 'OH'       
            389: 'KY'         
            25229: 'TX' 
        upc_ids:
            1600027527: 'cold_cereal'
            3800031838: 'cold_cereal'
    ...
    ...
```

##### model parameters

For the XGBoost model, there are a few parameters that need to be specified:

- `window_size` is used to create the number of lagged values as input for the model. A value of 52 will create 52 lags of the target column. 
- `avg_units` is used to create rolling averages using the lag-1 column to avoid leakage. It represents a list of rolling average features. A value of 2 will create a two time steps rolling-average, while a value of 16 will create a rolling-average of 16 time steps. Each will represent a column in the dataset.
- `gtrends_window_size` is used to cerate the number of lagged values for the google trends series. Each google trend series will be created usign this value.
- `search_iter` is used to specify how many rounds of hyperparameter search to perform using Hyperopt.

For the LSTM model, there are different parameters that need to be specified:

- `window_size` is used to create the number of lagged values as input for the model. A value of 52 will create 52 lags of the target column. 
- `gtrends_window_size` is used to cerate the number of lagged values for the google trends series. This value must be the same as the window_size as the LSTM expects the same number of dimensions for each feature.
- `dropout` is a hyperparameter that is specified in advance. It is used to reduce overfitting but could be included in the hyperparameter search if desired.
- `units_strategy` is used to determine how to select the number of hidden units for each LSTM layer. A stable strategy will keep the number of units constant accross layers. A decrease strategy will halve the number of units per layer. For a three layer model with initial number of units set to 50, the stable strategy will assign 50 units for each layer while the decrease strategy will set 50 for the first layer, 25 for the second layer and 16 for the third layer.
- `optimizers` the optimizer to use. 
- `loss` the loss to use.
- `search_iter` the number of random search iterations to perform during hyperparameter tunning.

Here is an example.  

```yaml
xgb:
    window_size: 52
    avg_units:
        - 2
        - 4
        - 8
        - 16
        - 26
    gtrends_window_size: 12
    search_iter: 100

lstm:
    window_size: 52
    gtrends_window_size: 52    # <- must match window_size
    dropout: 0.1
    units_strategy: 'decrease' # <- options are {decrease, stable}
    optimizers: 'adam'
    loss: 'mape'
    search_iter: 20
```

### Experiment Setup

After exploring the Brazilian e-commerce and the Breakfast at the Frat datasets. the product categories and products to generate forecasts for are selected.

The prediction task for the Brazilian e-commerce dataset is to forecast the weekly number of sales transactions by product category. The scope of sales transactions from the Brazilian dataset are limited to the Sao Paolo region and for the top 7 selling product categories. Hence, for the Brazilian e-commerce dataset, each forecasting model (SARIMA, FBProphet, XGBoost, LSTM) is run 7 times, once for each product category.

The prediction task for Breakfast at the Frat dataset is to forecast the weekly number of units sold of 4 items across 3 stores. Therefore, for the Breakfast at the Frat dataset, each forecasting model (SARIMA, FBProphet, XGBoost, LSTM) is run for each product and store combination. In other words, there are a total of 48 models run for the Breakfast at the Frat dataset. The data used from the Breakfast at the Frat dataset include sales history, promotional, product, manufacturer and store information. 


| ![Slide1](https://user-images.githubusercontent.com/39706513/101856789-05686d00-3b34-11eb-847f-e7037476d0c6.JPG) | 


TODO: explain experimental setup here (take from report)


### Models

- SARIMA (Baseline)
- FB Prophet (Baseline)
- XGBoost
- LSTM
