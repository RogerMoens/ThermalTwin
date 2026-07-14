from abc import ABC, abstractmethod

class BasePredictor(ABC):

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, folder):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass