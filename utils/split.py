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

        X_train, X_val, X_test = self._slice(X, train_end, val_end)
        y_train, y_val, y_test = self._slice(y, train_end, val_end)

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )

    @staticmethod
    def _slice(data, train_end, val_end):
        """Chronologically slice a DataFrame/Series or a plain ndarray."""

        if hasattr(data, "iloc"):
            return data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]

        return data[:train_end], data[train_end:val_end], data[val_end:]
