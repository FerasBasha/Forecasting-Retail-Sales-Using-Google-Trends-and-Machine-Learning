import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from tqdm import tqdm
import warnings
import os
from pathlib import Path
warnings.filterwarnings("ignore")

current_dir = Path.cwd()
proj_path = current_dir.parent.parent
# make the code in src available to import in this notebook
import sys
sys.path.append(os.path.join(proj_path,'src'))

# Custom imports
from metrics import mean_absolute_percentage_error


class SklearnSarima(BaseEstimator):
    '''
    Scikit-Learn wrapper for training a Sarima model
    using package sm.tsa.statespace.SARIMAX. To access the 
    original functions of a SARIMAX model, use model.X where X is the 
    function you wish to use.

    Args:
        x_series (np.array): The x_series is used for input to the model
        param (tuple): The param arguments are used for specifying pdq values
        param_seasonal (tuple): The param_seasonal arguments are used for specifying spdq values

    Attributes:
        x_series (np.array): The x_series is used for input to the model
        param (tuple): The param arguments are used for specifying pdq values
        param_seasonal (tuple): The param_seasonal arguments are used for specifying spdq values
    '''
    def __init__(self, x_series, param: tuple=None, 
                 param_seasonal: tuple=None):
        
        self.x_series = x_series
        self.param_pdq = param
        self.param_spdq = param_seasonal

        
    def fit_predict(self, y_values):
        
        predictions = []
        
        for i in range(len(y_values)):
            # fit with x[1:] + y[:1] step, predict 
            # predict 1 step
            # The model is already defined, we don't change it.
            predictions.append(self.predict(1))
            self.set_xseries(np.concatenate([self.x_series[i:],y_values[:i]]))
            self.fit()
            
        return predictions
    

    def fit(self) -> None:
        '''
        Fit a SARIMAX model. The x_series, param_pdq and param_spdq must be set first.        
        '''
        
        self.model = sm.tsa.statespace.SARIMAX(self.x_series,
                                               order=self.param_pdq,
                                               seasonal_order=self.param_spdq,
                                               enforce_qqstationarity=False,
                                               enforce_invertibility=False)

        self.model = self.model.fit()


    def predict(self, forecast_period) -> np.array:
        '''
        Make predictions for the desired length.
        
        Args:
            forecast_period (int): The forecast_period is used to indicate
            how many steps to forecast.
            
        Returns:
            predictions (list): The forecasted values
        '''
        
        predictions = self.model.get_forecast(steps=forecast_period)
        return predictions.predicted_mean


    def fit_best_params(self, y_series, search_params=list) -> None:
        '''
        Using a grid-search approach, fit a SARIMA model and evaluate on
        a validation set. Select the best set of parameters and train 
        model with it.
        
        Args:
            y_series (np.array): The y_series is used as the validation set
            search_params (list): The search_params is used as the list of 
            search terms to fit and evaluate the model. The list takes the
            form of :  [((p, d, q), (s, p, d, q)), 
                        ((p, d, q), (s, p, d, q)), ...]
        
        '''
        
        results_preds = []
        for param in tqdm(search_params, desc="Finding best parameters"):
            # Fit a model with the best params    
            self.set_params({'pdq':param[0], 'spdq':param[1]})
            
            try:
                self.fit()
                predictions = self.model.get_forecast(steps=y_series.size)
                mape_result = mean_absolute_percentage_error(y_series, predictions.predicted_mean)

            except:
                self.model.aic = np.nan
                mape_result = np.nan

            # Store configuration and results and then append to the results_preds data frame
            # AIC stands for Akaike information criterion and can be used to estimate the quality of the models
            # We can use both AIC and MSE 
            row = {'pdq': param[0],
                   'spdq': param[1],
                   'Aic': self.model.aic,
                   'Mse': mape_result}

            results_preds.append(row)
            
        self.search_results = pd.DataFrame(results_preds).dropna()
        
        # Select the row with the smallest AIC
        best_params = self.search_results.sort_values(by='Aic').iloc[0]
        
        # Fit a model with the best params    
        self.set_params({'pdq':best_params['pdq'],
                         'spdq':best_params['spdq']})
        self.set_xseries(np.concatenate((self.x_series, y_series)))
        self.fit()


    def set_params(self, params: dict) -> None:
        '''
        Following best practices, this function sets the parameters
        of the model so that none of the methods are able to directly
        change the global variables.
        
        Args: 
            params (dict): The params are used to set the parameters of the model
        '''
        self.param_pdq = params['pdq']
        self.param_spdq = params['spdq']

        
    def set_xseries(self, x_series) -> None:
        self.x_series = x_series
        

    def get_params(self) -> dict:
        '''
        Get the parameters.
        
        Returns: 
            params (dict): The params that are used by the model
        '''        

        return {'pdq':self.param_pdq,
                'spdq':self.param_spdq}
