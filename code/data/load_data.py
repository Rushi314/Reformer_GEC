import csv
import json
import torch

def dataset_reader(name):
    with open(f'drive/Shareddrives/CS685_Reformer_GEC/Data/FCE/fce_out/{name}.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            yield row

def load_del_data():
    datasets = []
    for name in ('train', 'val'):
        dataset = []
        for row in dataset_reader(name):
            x = row[1]
            y = torch.tensor(eval(row[4]), dtype=float)

            assert len(x) == len(y)

            dataset.append((x, y))
        datasets.append(dataset)

    return tuple(datasets)

def load_ins_data():
    datasets = []
    for name in ('train', 'val'):
        dataset = []
        for row in dataset_reader(name):
            x = row[3]
            y = [None if len(item) == 0 else item for item in eval(row[5])]

            assert len(str.encode(x)) + 1 == len(y)

            dataset.append((x, y))
        datasets.append(dataset)

    return tuple(datasets)