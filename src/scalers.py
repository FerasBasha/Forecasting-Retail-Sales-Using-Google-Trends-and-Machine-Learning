from sklearn.base import BaseEstimator
import numpy as np


class MinMaxScalerDf(BaseEstimator):
    """ Scale the whole data, not just one column 
        using the min and max values.
    """
    def __init__(self):
        pass
    
    def fit_transform(self, tensor_3d):
        self.min_ = np.nanmin(tensor_3d)
        self.max_ = np.nanmax(tensor_3d)
        return (tensor_3d - self.min_) / (self.max_ - self.min_)
        
    def transform(self, tensor_3d):
        return (tensor_3d - self.min_) / (self.max_ - self.min_)
    
    def inverse_transform(self, tensor_3d):
        return tensor_3d * (self.max_ - self.min_) + self.min_


class NormalizeScalerDf(BaseEstimator):
    """ Normalize the whole data, not just one column with 
        mean 0 and standard deviation of 1.
    """
    def __init__(self):
        pass
    
    def fit_transform(self, tensor_3d):
        self.mean_ = np.nanmean(tensor_3d)
        self.std_ = np.nanstd(tensor_3d)
        return (tensor_3d - self.mean_) / self.std_
        
    def transform(self, tensor_3d):
        return (tensor_3d - self.mean_) / self.std_
    
    def inverse_transform(self, tensor_3d):
        return (tensor_3d * self.std_) + self.mean_