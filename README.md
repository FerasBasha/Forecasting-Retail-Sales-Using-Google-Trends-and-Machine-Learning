## Forecasting Retail Sales Using Google Trends and Machine Learning 

| <img width="736" alt="KelloggsFrostedFlakes in Texas" src="https://user-images.githubusercontent.com/39706513/103062604-b6332b00-457c-11eb-8928-4df8e6eb2adc.png"> | 
|:--:| 
| *Example of Predictions using XGBoost for Kellogg's Frosted Flakes Sales in Texas (Breakfast At The Frat)* |

### Description 

The primary objective of this experiment is to propose a methodological framework to incorporate external data, in particular from Google Trends, in retail sales forecasting by leveraging modern machine learning techniques. 

In order to investigate the predictive power of [Google Trends](https://trends.google.com/trends/?geo) we use the [Brazilian e-commerce by Olist](https://www.kaggle.com/olistbr/brazilian-ecommerce) as well as the [Breakfast at the Frat by dunnhumby](https://www.dunnhumby.com/source-files/) public datasets, to conduct a quantitative experiment in which we compare the predictive performance on sales forecasts of the following models: a) the Seasonal Autoregressive Integrated Moving Average (SARIMA) model, b) The Facebook Prophet tool (FBProphet), c) The Extreme Gradient Boosting algorithm (XGBoost), and d) a recurrent neural network with long short-term memory (LSTM). To measure forecasting accuracy, various performance metrics are used, and the performance of all forecasting models is benchmarked against a naïve model. 

The source code of the experiment is made available to the public and can be adapted in future projects.


### Prediction Task  

- The prediction task for the Brazilian e-commerce dataset is to forecast the weekly number of sales transactions by product category. The scope of sales transactions from the Brazilian e-commerce dataset are limited to the Sao Paolo region and for the top 7 selling product categories. Thus, the Brazilian e-commerce dataset is split into 7 separate datasets and each forecasting model (SARIMA, FBProphet, XGBoost, LSTM) is trained and tested 7 times, once for each product category

- The prediction task for Breakfast at the Frat dataset is to forecast the weekly number of units sold of 4 items across 3 stores. Therefore, the Breakfast at the Frat dataset is split into 12 separate datasets and each forecasting model (SARIMA, FBProphet, XGBoost, LSTM) is trained and tested 12 times, once for each product and store combination. The data used from the Breakfast at the Frat dataset include sales history, promotional, product, manufacturer and store information. 

### Data Input and Perfomrance Comparison Framework

<div align="center">
</div>

| ![Capture](https://user-images.githubusercontent.com/39706513/102944008-9110c080-4487-11eb-9bc8-2339261d1e39.PNG) | 
|:--:|

### Experiment Setup 

- The experiment utilizes configuraiton & parameter files to pre-process data and determine parameter values required to run the forecasting models. 

<div align="center">
  Experiment Conceptual Diagram
</div>

|![ExperimentDesign_MSc](https://user-images.githubusercontent.com/39706513/101991375-4bc7e400-3c7a-11eb-968f-00dbf5d85617.png) | 
|:--:|
| *[MLflow](https://mlflow.org/docs/latest/tracking.html) is used to track parameters and performance metrics* |


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
