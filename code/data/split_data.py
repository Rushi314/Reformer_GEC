import csv
import numpy as np

seed = 17
train_fraction = 0.9

def split_train_val():
    data = []
    with open('drive/Shareddrives/CS685_Reformer_GEC/Data/FCE/fce_out/FinalDataUpdated.csv', 'r') as train_file:
        reader = csv.reader(train_file)
        header = next(reader)
        for row in reader:
            # Discard data containing unsupported characters
            if len(str.encode(row[1])) == len(row[1]) and len(str.encode(row[2])) == len(row[2]):
                data.append(row)

    rng = np.random.default_rng(seed)
    rng.shuffle(data)

    split = int(len(data) * train_fraction)
    train_data = data[:split]
    val_data = data[split:]
    
    datasets = (('train', train_data), ('val', val_data))

    for name, dataset in datasets:
        with open(f'drive/Shareddrives/CS685_Reformer_GEC/Data/FCE/fce_out/{name}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(dataset)

if __name__ == '__main__':
    split_train_val()