print("Importing your stupid garbage")
import os
import pickle
import numpy as np
import pandas as pd
import sklearn.linear_model
import random
from tqdm import tqdm

def add_weather(data, weather_data):
    """Merge weather data into the train and test data."""
    for idx, row in data.iterrows():
        ts = row["timestamp"]
        weather_row_idx = np.argmin(np.abs(weather_data["timestamp"] - ts))
        for col in weather_data.columns[1:]:
            data.loc[idx, col] = weather_data.loc[weather_row_idx, col]
    return data

def to_batches(data, splits):
    """Split the data into batches according to a given array of ranges."""
    batches = []
    for split in splits:
        batches.append(data.iloc[split])
    return batches

def row_by_time(data, time):
    """Get a row whose timestamp is as close as possible to time."""
    nearest = data.iloc[np.argmin(np.abs(data["timestamp"] - time))]
    return nearest

def count_h_ago(data, train_data, hours, column=None):
    """Number of bikes X hours ago."""
    if column is None:
        return np.array([
            row_by_time(train_data, ts)
            for ts in data["timestamp"] - pd.Timedelta(hours=hours)
        ])
    else:
        return np.array([
            row_by_time(train_data, ts)[column]
            for ts in data["timestamp"] - pd.Timedelta(hours=hours)
        ])

meta_data = pd.read_csv("data/bicikelj_metadata.csv", sep="\t")
meta_data = meta_data.set_index("postaja")
if not (
    os.path.exists("data/train_data_preprocessed.pkl") and
    os.path.exists("data/test_data_preprocessed.pkl") and
    os.path.exists("data/splits.pkl")
):
    print("Preprocessing data ...")

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

    # add 1h ago, 2h ago and 3h ago columns
    h_ago1 = pd.DataFrame(count_h_ago(train_data, train_data, 1))
    h_ago2 = pd.DataFrame(count_h_ago(train_data, train_data, 2))
    h_ago3 = pd.DataFrame(count_h_ago(train_data, train_data, 3))

    h_ago1_test = pd.DataFrame(count_h_ago(test_data, train_data, 1))
    h_ago2_test = pd.DataFrame(count_h_ago(test_data, train_data, 2))
    h_ago3_test = pd.DataFrame(count_h_ago(test_data, train_data, 3))

    h_ago1.columns = train_data.columns.values + "_1h_ago"
    h_ago2.columns = train_data.columns.values + "_2h_ago"
    h_ago3.columns = train_data.columns.values + "_3h_ago"

    h_ago1_test.columns = test_data.columns.values + "_1h_ago"
    h_ago2_test.columns = test_data.columns.values + "_2h_ago"
    h_ago3_test.columns = test_data.columns.values + "_3h_ago"

    train_data = pd.concat([train_data, h_ago1], axis='columns')
    train_data = pd.concat([train_data, h_ago2], axis='columns')
    train_data = pd.concat([train_data, h_ago3], axis='columns')

    test_data = pd.concat([test_data, h_ago1_test], axis='columns')
    test_data = pd.concat([test_data, h_ago2_test], axis='columns')
    test_data = pd.concat([test_data, h_ago3_test], axis='columns')
    
    # divide columns by the station capacity
    # for station_name, row in meta_data.iterrows():
    #     train_data.loc[:, station_name] /= row["total_space"]
    #     test_data.loc[:, station_name] /= row["total_space"]

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
    pickle.dump(splits, open("data/splits.pkl", "wb"))

    # interpolate missing values in rain column
    weather_data["rain"] = weather_data["rain"].interpolate()

    # merge weather data into train and test data
    train_data = add_weather(train_data, weather_data)
    test_data = add_weather(test_data, weather_data)

    # save to file
    pickle.dump(train_data, open("data/train_data_preprocessed.pkl", "wb"))
    pickle.dump(test_data, open("data/test_data_preprocessed.pkl", "wb"))

train_data = pickle.load(open("data/train_data_preprocessed.pkl", "rb"))
test_data = pickle.load(open("data/test_data_preprocessed.pkl", "rb"))
splits = pickle.load(open("data/splits.pkl", "rb"))





def prepare_data(data, train_data, column):
    """Construct features from data."""
    total_space = meta_data.loc[column, "total_space"]
    X = np.stack([
        # data.join(train_data.set_index("timestamp"), on="timestamp", how="left", rsuffix="aaaaa")[column + "_1h_ago"].values,
        # data.join(train_data.set_index("timestamp"), on="timestamp", how="left", rsuffix="aaaaa")[column + "_2h_ago"].values,
        # count_h_ago(data, train_data, 1, column),
        # count_h_ago(data, train_data, 2, column),
        # [total_space for _ in range(len(data))],
        data[column + "_1h_ago"].values,
        # data[column + "_1h_ago"].values == 0,
        # data[column + "_1h_ago"].values == total_space,
        data[column + "_2h_ago"].values,
        data[column + "_2h_ago"].values < data[column + "_1h_ago"].values,
        data[column + "_3h_ago"].values < data[column + "_2h_ago"].values,
        # data.join(train_data.set_index("timestamp"), on="timestamp", how="left", rsuffix="aaaaa")[column + "_3h_ago"].values,
        # hour,
        data["timestamp"].dt.month.values + 1 < 9,  # pocitnice
        # data["timestamp"].dt.month.values,
        data["timestamp"].dt.weekday.isin([5, 6]),  # vikend
        data["timestamp"].dt.weekday.isin([0, 1, 2, 3, 4]),  # delovnik

        data["timestamp"].dt.hour.isin([0, 1, 2, 3, 4, 5]), # noc
        data["timestamp"].dt.hour.isin([6, 7, 8, 9, 10, 11]), # dopoldne
        data["timestamp"].dt.hour.isin([12, 13, 14, 15, 16, 17]), # popoldne
        data["timestamp"].dt.hour.isin([18, 19, 20, 21, 22, 23]), # vecer
        # np.logical_and(6 < hour, hour < 17),
        # np.logical_and(10 < hour, hour < 15),
        data["temperature"].values > 30,
        # data["temperature"].values,
        data["temperature"].values < 10,
        # data["humidity"].values,
        data["rain"].values,
        data["rain"].values == 0,
        data["rain"].values > 0.3,
        data["rain"].values > 1,
    ], axis=1)

    y = data[column].values

    # center X
    # X = X - X.mean()

    return X, y

def train_predict(train_data, test_data, column):
    clf = sklearn.linear_model.LinearRegression()

    X_train, y_train = prepare_data(train_data, train_data, column)
    X_test, y_test = prepare_data(test_data, train_data, column)

    n_poly = 2
    poly_features_train = sklearn.preprocessing.PolynomialFeatures(n_poly, interaction_only=True).fit_transform(X_train, y_train)
    poly_features_test = sklearn.preprocessing.PolynomialFeatures(n_poly, interaction_only=True).fit_transform(X_test)

    clf.fit(poly_features_train, y_train)
    predictions = clf.predict(poly_features_test)
    return predictions


# train model
for column in [test_data.columns[1]]:
    print(f"Predicting column {column}")
    # for column in tqdm(meta_data.index):
    if True:
        batches = to_batches(train_data, splits)

        # eval data is zero and one hour before end of every batch
        eval_data = pd.concat([
            pd.concat([
                batch.iloc[-1],
                row_by_time(batch, batch.iloc[-1].timestamp - pd.Timedelta(hours=1))
            ], axis=1).T
            for batch in batches
        ])

        # drop last 2 hours of every batch's training data
        for i, batch in enumerate(batches):
            last_row = row_by_time(batch, batch.iloc[-1].timestamp - pd.Timedelta(hours=2))
            batches[i] = batch[batch.timestamp < last_row.timestamp]

        train_data = pd.concat(batches)

        predictions = train_predict(train_data, eval_data, column)
        print("MAE:", np.mean(np.abs(predictions - eval_data[column].values)))
    else:
        predictions = train_predict(train_data, test_data, column)
        predictions[predictions < 0] = 0
        test_data[column] = predictions
        # print(predictions)

# write predictions to file
columns_to_keep = ["timestamp"] + list(meta_data.index)
test_data = test_data[columns_to_keep]
test_data.to_csv("predictions.csv", index=False)
print("Saved")

