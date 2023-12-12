from sklearn.preprocessing import StandardScaler as StandardScaler_
from sklearn.preprocessing._data import check_is_fitted
import torch


class StandardScaler(StandardScaler_):

    def fit(self, *args, **kwargs):
        ret = super().fit(*args, **kwargs)
        if self.with_mean:
            self.mean_tensor = torch.from_numpy(self.mean_)
        if self.with_std:
            self.scale_tensor = torch.from_numpy(self.scale_)
        return ret
    
    def mean(self, X=None):
        return (self.mean_tensor.to(X.device) if torch.is_tensor(X) else self.mean_)
        
    def scale(self, X=None):
        return (self.scale_tensor.to(X.device) if torch.is_tensor(X) else self.scale_)

    def transform(self, X, copy=None):
        check_is_fitted(self)

        if self.with_mean:
            X = X - self.mean(X)
        if self.with_std:
            X = X / self.scale(X)
        return X

    def inverse_transform(self, X, copy=None):
        check_is_fitted(self)

        if self.with_std:
            X = X * self.scale(X)
        if self.with_mean:
            X = X + self.mean(X)
        return X
