from sklearn.preprocessing import StandardScaler as StandardScaler_
from sklearn.preprocessing._data import check_is_fitted

class StandardScaler(StandardScaler_):
    def transform(self, X, copy=None):
        check_is_fitted(self)

        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        check_is_fitted(self)

        if self.with_std:
            X = X * self.scale_
        if self.with_mean:
            X = X + self.mean_
        return X
