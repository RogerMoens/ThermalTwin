class DatasetSplitter:

    def __init__(
        self,
        train_size=0.70,
        val_size=0.15,
    ):
        self.train_size = train_size
        self.val_size = val_size

    def split(self, X, y):

        n = len(X)

        train_end = int(self.train_size * n)
        val_end = int((self.train_size + self.val_size) * n)

        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]

        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )