import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogAtt(BaseEstimator, TransformerMixin):
    def __init__(self, tsd_loc, chl_loc, rs_loc, sp_loc) -> None:
        self.tsd = tsd_loc
        self.chl = chl_loc
        self.rs = rs_loc
        self.sp = sp_loc

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X[~(X[:,-1] == 3),:]
        X = X[~(X[:,-1] == 8),:]

        X_l = X
        X_l[:, self.tsd] = np.log(X[:, self.tsd])
        X_l[:, self.chl] = np.log(X[:, self.chl])
        X_l[:, self.rs] = np.log(X[:, self.rs])
        X_l[:, self.sp] = np.log(X[:, self.sp])
        
        return X_l[:, :-1], X_l[:, -1]
