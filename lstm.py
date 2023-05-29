import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

def add_weather(data, weather_data):
    for idx, row in data.iterrows():
        ts = row["timestamp"]
        weather_row_idx = np.argmin(np.abs(weather_data["timestamp"] - ts))
        for col in weather_data.columns[1:]:
            data.loc[idx, col] = weather_data.loc[weather_row_idx, col]
    return data

class Predictor:
    def __init__(self, column, lr=0.001, n_epochs=300):
        self.column = column
        self.lr = lr
        self.n_epochs = n_epochs

        # LSTM() returns tuple of (tensor, (recurrent state))
        class extract_tensor(nn.Module):
            def forward(self, x):
                # Output shape (batch, features, hidden)
                tensor, _ = x
                # Reshape shape (batch, hidden)
                return tensor

        self.model = nn.Sequential(
            nn.LSTM(10, 10, batch_first=True),
            extract_tensor(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

        self.loss_fn = nn.L1Loss()  # MAE
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def prepare_data(self, data):
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
            weekday < 5,
            data["temperature"].values,
            data["humidity"].values,
            data["rain"].values,
        ], axis=1)
        y = data[self.column].values
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y

    def train(self, train_data_batches):
        print("Begin training!")
        try:
        #for epoch in range(self.n_epochs):
            epoch = -1
            losses = []
            while True:
                epoch += 1
                for batch in train_data_batches:
                    Xbatch, ybatch = self.prepare_data(batch)

                    y_pred = self.model(Xbatch)
                    loss = self.loss_fn(y_pred, ybatch.unsqueeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                true_loss = self.loss_fn(y_pred[-2:], ybatch[-2:].unsqueeze(1))
                losses.append(true_loss.item()*20)
                print(f'Epoch {epoch+1}; loss = {true_loss.item()*20}')
                if epoch % 50 == 0:
                    plt.plot(losses)
                    plt.savefig(f'losses_{epoch}.png')
        except KeyboardInterrupt:
            print("Done training!")

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
meta_data = pd.read_csv("data/bicikelj_metadata.csv", sep="\t")
meta_data = meta_data.set_index("postaja")

print("Preprocessing ...")

# divide columns by the station capacity
for station_name, row in meta_data.iterrows():
    train_data.loc[:, station_name] /= row["total_space"]
    test_data.loc[:, station_name] /= row["total_space"]

# determine dataset splits
splits = [None for _ in range(len(test_data) // 2)]
split_idx = 0
split_start = 0
for row in train_data.itertuples():
    if row.timestamp > test_data.iloc[2*split_idx].timestamp:
        splits[split_idx] = range(split_start, row.Index)
        split_idx += 1
        split_start = row.Index
splits[-1] = range(split_start, len(train_data))

# interpolate missing values in rain column
weather_data["rain"] = weather_data["rain"].interpolate()

# merge weather data into train and test data
train_data = add_weather(train_data, weather_data)
test_data = add_weather(test_data, weather_data)

# train model
for column in [test_data.columns[1]]:  # test_data.columns[1:]:
    column = train_data.columns[1]
    print(f"Predicting column {column}")
    predictor = Predictor(column)

    def to_batches(data, splits):
        batches = []
        for split in splits:
            batches.append(data.iloc[split])
        return batches

    batches = to_batches(train_data, splits)
    if False:
        eval_data = pd.concat([batch[-2:] for batch in batches])
        train_data = [batch[:-2] for batch in batches]
        predictor.train(train_data)
        predictor.predict(eval_data)
    else:
        predictor.train(batches)
        predictions = predictor.predict(test_data) * meta_data.loc[column].at["total_space"]
        test_data[column] = predictions.round().squeeze().numpy()
        print(predictions)

# write predictions to file
test_data.drop(columns=["temperature", "humidity", "rain"], inplace=True)
test_data.to_csv("predictions.csv", index=False)
print("Saved")

