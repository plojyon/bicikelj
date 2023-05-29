import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def add_weather(data, weather_data):
    for idx, row in data.iterrows():
        ts = row["timestamp"]
        weather_row_idx = np.argmin(np.abs(weather_data["timestamp"] - ts))
        for col in weather_data.columns[1:]:
            data.loc[idx, col] = weather_data.loc[weather_row_idx, col]
    return data

class Predictor:
    def __init__(self, column, lr=0.001, n_epochs=10, batch_size=10):
        self.column = column
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.model = nn.Sequential(
            nn.Linear(9, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        self.model.apply(init_weights)

        self.loss_fn = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def prepare_data(self, data):
        print("Preparing data")
        count_before = data[self.column].shift(1).fillna(0).values
        count_after = data[self.column].shift(-1).fillna(0).values
        time_before = data["timestamp"].shift(1).fillna(data["timestamp"].values[0])
        time_after = data["timestamp"].shift(-1).fillna(data["timestamp"].values[-1])
        hour = data["timestamp"].dt.hour.values + data["timestamp"].dt.minute.values / 60
        weekday = data["timestamp"].dt.weekday.values # 0 is Monday, 6 is Sunday
        X = np.stack([
            count_before,
            count_after,
            time_before.dt.hour.values + time_before.dt.minute.values / 60,
            time_after.dt.hour.values + time_after.dt.minute.values / 60,
            hour,
            weekday,
            data["temperature"].values,
            data["humidity"].values,
            data["rain"].values,
        ], axis=1)
        y = data[self.column].values
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        print("Data prepared")
        print(X, y)
        return X, y

    def train(self, train_data):
        X, y = self.prepare_data(train_data)

        print("Begin training!")
        for epoch in range(self.n_epochs):
            for i in range(0, len(X), self.batch_size):
                Xbatch = X[i:i+self.batch_size]
                y_pred = self.model(Xbatch)
                ybatch = y[i:i+self.batch_size]
                loss = self.loss_fn(y_pred, ybatch.unsqueeze(1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch}; loss = {loss}')

    def predict(self, test_data):
        X, y = self.prepare_data(test_data)
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred


print("Reading data ...")

train_data = pd.read_csv("data/bicikelj_train.csv")
train_data["timestamp"] = [
    pd.to_datetime(ts).tz_localize(None)
    for ts in train_data["timestamp"].values
]
test_data = pd.read_csv("data/bicikelj_test.csv")
test_data["timestamp"] = [
    pd.to_datetime(ts).tz_localize(None)
    for ts in test_data["timestamp"].values
]
weather_data = pd.read_csv("data/weather.csv")
weather_data["timestamp"] = [
    pd.to_datetime(ts).tz_localize(None)
    for ts in weather_data["timestamp"].values
]

# interpolate missing values in rain column
weather_data["rain"] = weather_data["rain"].interpolate()

# merge weather data into train and test data
print("Adding weather data ...")
train_data = add_weather(train_data, weather_data)
test_data = add_weather(test_data, weather_data)

# train model
print(f"Predicting column {train_data.columns[1]}")
predictor = Predictor(train_data.columns[1])

if False:
    eval_data = train_data.sample(frac=0.1)
    train_data = train_data.drop(eval_data.index)
    predictor.train(train_data)
    predictor.predict(eval_data)
else:
    predictor.train(train_data)
    p = predictor.predict(test_data)
    print(p)

