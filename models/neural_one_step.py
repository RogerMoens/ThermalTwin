class NeuralOneStepPredictor(BasePredictor):

    def __init__(self):

        self.scaler = StandardScaler()

        self.model = Sequential([
            Input(shape=(102,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1)
        ])