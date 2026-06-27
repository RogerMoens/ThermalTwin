class LinearOneStepPredictor(BasePredictor):

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()

    def fit(self, X_train, y_train, **kwargs):

        X = self.scaler.fit_transform(X_train)
        self.model.fit(X, y_train)

    def predict(self, X):

        X = self.scaler.transform(X)
        return self.model.predict(X)

    def save(self, folder):

        joblib.dump(self.model, folder / "linear.pkl")
        joblib.dump(self.scaler, folder / "scaler.pkl")

    @classmethod
    def load(cls, folder):

        obj = cls()
        obj.model = joblib.load(folder / "linear.pkl")
        obj.scaler = joblib.load(folder / "scaler.pkl")
        return obj