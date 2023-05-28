import pandas as pd

train = pd.read_csv("data/bicikelj_train.csv")
train["timestamp"] = [pd.to_datetime(ts).tz_localize(None) for ts in train["timestamp"].values]

test = pd.read_csv("data/bicikelj_test.csv")
test["timestamp"] = [pd.to_datetime(ts).tz_localize(None) for ts in test["timestamp"].values]

# predict with mean value
colmeans = train.mean(axis=0, numeric_only=True)
test.iloc[:, 1:] = colmeans
test.to_csv("station_mean.csv", sep=",", index=False)