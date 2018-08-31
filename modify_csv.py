# The avg_train_times.py program outputs a large csv of all data that
# is possibly relevant.  This file reads that csv and does some basic
# column removal and sorting to make the data easier to read into
# matlab and analyze.
# The keep_cols list must be set manually by observing the format
# string in avg_train_times.py and deciding which vars are relevant.

import csv
in_file = "output/cpu_usage_comp/all_data.csv"
out_file = "output/cpu_usage_comp/all_data_trimmed.csv"
# keep_cols = [0, 1, 6, 3, 4, 5, 11]
# keep_cols = [0, 1, -1, -2, -3]
keep_cols = [0, 1, -2, 7, 11]
data = []

with open(in_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append([row[index] for index in keep_cols])


for row in data:
    if len(row[2]) == 1:
        row[2] = '0' + row[2]

# for row in data:
#     if len(row[3]) == 1:
#         row[3] = '0' + row[3]

data.sort()
# data[:6] = sorted(data[:6], lambda x:x[1])

with open(out_file, 'w') as f:
    writer = csv.writer(f)
    for row in data: writer.writerow(row)
