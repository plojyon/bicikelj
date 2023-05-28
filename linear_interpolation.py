import csv
import random
import pandas as pd
import numpy as np

def predict(case, train_data):
    """Predict the value of case using train_data."""
    # find the closest row in train_data according to case[0]
    ts = case[0].to_numpy()
    closest_idx = np.argmin(np.abs(train_data["timestamp"].values - ts))
    
    # find nearest lower and upper neighbour
    lower_idx = closest_idx
    upper_idx = closest_idx
    if train_data.iloc[closest_idx, 0] < ts:
        upper_idx += 1
        if upper_idx > len(train_data) - 1:
            upper_idx = closest_idx
    else:
        lower_idx -= 1
        if lower_idx < 0:
            lower_idx = closest_idx

    lower = train_data.iloc[lower_idx, :]
    upper = train_data.iloc[upper_idx, :]

    # linearly interpolate
    if upper[0] != lower[0]:
        alpha = (ts - lower[0]) / (upper[0] - lower[0])
    else:
        alpha = 1
    predicted = (1-alpha) * lower[1:] + alpha * upper[1:]

    return np.array([ts] + [np.round(i) for i in predicted])


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


if False:
    # evaluate performance
    eval_data = train_data.sample(frac=0.1)
    train_data = train_data.drop(eval_data.index)
    rmse = 0
    mae = 0
    for idx, row in eval_data.iterrows():
        prediction = predict(row, train_data)
        rmse += (prediction[1:] - row[1:]) ** 2
        mae += np.abs(prediction[1:] - row[1:])

    rmse = (rmse / len(eval_data)) ** 0.5
    mae = mae / len(eval_data)
    # print(f"RMSE: {rmse}")
    print(f"RMSE mean: {np.mean(rmse):.4f}")
    # print(f"MAE: {mae}")
    print(f"MAE mean: {np.mean(mae):.4f}")
else:
    # predict
    for idx, case in test_data.iterrows():
        prediction = predict(case, train_data)
        test_data.iloc[idx, 1:] = prediction[1:]

    test_data.set_index("timestamp", inplace=True, drop=True)
    test_data.to_csv("predictions.csv", sep=",")
