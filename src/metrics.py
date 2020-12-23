import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.metrics import r2_score
import mlflow
  
#Function selected to evaluate performance to allow for comparison with the other baseline models in this project
#Daily sales values may have a value of 'zero' on a given day. That is why the term '+ 1' is added to the function
def mean_absolute_percentage_error(y_true,y_pred):

    y_true = np.array(y_true) + 1 
    y_pred = np.array(y_pred) + 1

    return np.mean(np.abs((y_true - y_pred) / y_true))


#https://help.sap.com/saphelp_scm700_ehp03/helpdata/en/76/487053bbe77c1ee10000000a174cb4/content.htm?no_cache=true
def weighted_absolute_percentege_error(y_true,y_pred):
    
    y_true = np.array(y_true) + 1e-6 
    y_pred = np.array(y_pred) + 1e-6
     
    return np.sum(np.abs(y_true - y_pred) * y_true) / np.sum(y_true)


#https://github.com/shang9922/Subway/blob/e81323387f8b5f1c306fc7bdfc29ffef5a12bce6/Exam.py
def weighted_mean_absolute_error(test, predict):
    #adding +1 to sales data becuase there maybe 0 in actuals
    test = np.array(test) + 1 
    predict = np.array(predict) + 1
    fenmu = max(test)
    rs = []
    for i in range(len(test)):
        if test[i] == 0:
            p = 1
        else:
            p = test[i]
        fenzi = (abs(test[i] - predict[i]))*p*p
        rs.append(float(fenzi)/fenmu)
    return np.mean(rs)

#https://www.kaggle.com/danspace/rossmann-store-sales-xgboost
#Below function is typically being used in Kaggle competitions for early stopping 
#def rmspe_xg(yhat, y):
    #y = y.get_label()
    #yhat = yhat
    #return "rmspe", rmspe(y,yhat)
def get_metrics(y_true, y_pred, param_prefix='validation'):
    """
    Calculate metrics and print them. If log=True, then a run_id should be passed to it ~32 charracters.
    The function will log the metrics to mlflow if log=True. A parameter
    prefix string should also be passed to it, to add to the metric name so it 
    doesn't overright when you log the parameters.
    
    Args
        y_true: real values
        y_pred: prediction values
        run_id: string, ~32 charactors long
        log: bool, <default False>
        param_prefix: string, id to add to the metric names <default 'validation'>
    Returns
        None
    """
    
    wape = weighted_mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2_metric = r2_score(y_true, y_pred)
    mape_metric = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {'wape':wape,
            'rmse':rmse,
            'r2':r2_metric,
            'mape':mape_metric,
            'mae':mae}
