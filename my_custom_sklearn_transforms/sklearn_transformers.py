from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class Resample(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.smote = SMOTE(random_state=1)

    def fit(self):
        return self

    def transform(self, X, y):
        return self.smote.fit_resample(x, y)
    
    