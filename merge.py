"""Merge data/bicikelj_train.csv and data/bicikelj_test.csv."""
import csv


with open('data/bicikelj_train.csv', 'r') as train_file:
    train_reader = csv.reader(train_file)
    train_data = list(train_reader)

with open('data/bicikelj_test.csv', 'r') as test_file:
    test_reader = csv.reader(test_file)
    test_data = list(test_reader)

headers = test_data[0] # == train_data[0]
train_data.pop(0)
test_data.pop(0)

# merge - combine both files, sort by timestamp
merged_data = test_data + train_data
merged_data.sort(key=lambda x: x[0])
merged_data.insert(0, headers)

with open('data/bicikelj_merged.csv', 'w') as merged_file:
    merged_writer = csv.writer(merged_file, lineterminator="\n")
    merged_writer.writerows(merged_data)


# split - for each train row, copy all train data and add one test row
# for i, test_case in enumerate(test_data):
#     merged_data = [test_case] + train_data
#     merged_data.sort(key=lambda x: x[0])
#     merged_data.insert(0, headers)

#     with open(f'data/test_cases/test_{i}.csv', 'w') as merged_file:
#         merged_writer = csv.writer(merged_file, lineterminator="\n")
#         merged_writer.writerows(merged_data)
