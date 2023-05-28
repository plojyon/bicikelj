import numpy as np
import pandas as pd

train = pd.read_csv("data/bicikelj_train.csv")
train["timestamp"] = [pd.to_datetime(ts).tz_localize(None) for ts in train["timestamp"].values]

test = pd.read_csv("data/bicikelj_test.csv")
test["timestamp"] = [pd.to_datetime(ts).tz_localize(None) for ts in test["timestamp"].values]

times = train["timestamp"].values
ptimes = test["timestamp"].values

for i, t in enumerate(ptimes):
    closest = np.argmin(np.abs(times - t))
    test.iloc[i, 1:] = train.iloc[closest, 1:]

test.to_csv("closest_time.csv", sep=",", index=False)