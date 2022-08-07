import pandas as pd
import numpy as np
import torch

def load_cardio_dataset(data_path, train_size=1776, test_size=350):
    df = pd.read_excel(io=data_path, sheet_name='Data', header=1)
    columns_to_keep = set(['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width',
                           'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'CLASS'])
    columns_to_delete = [col for col in df.columns if col not in columns_to_keep]
    df.drop(columns_to_delete, axis=1, inplace=True) # Keep the features + class
    df.drop(df.tail(3).index, inplace=True) # Drop last 3 rows
    df = df.sample(frac=1)
    list_data_features = df[[col for col in df.columns if col != 'CLASS']].to_numpy()
    
    labels = df['CLASS'].to_numpy()-1.
    n_classes = len(set(labels))
    labels = torch.LongTensor(labels).unsqueeze(1)
    labels = torch.nn.functional.one_hot(labels, num_classes=n_classes).float().squeeze(1)
    data = torch.FloatTensor(list_data_features)
    
    train_labels, train_data = labels[:train_size], data[:train_size]
    test_labels, test_data = labels[train_size:train_size+test_size], data[train_size:train_size+test_size]
    
    letter_dataset = {
        'train': (train_data, train_labels),
        'test': (test_data, test_labels),
    }
    
    assert train_size+test_size <= len(df)
    return letter_dataset

class Cardio_Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        super().__init__()
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        inp = self.inputs[i]
        tar = self.targets[i]
        return inp, tar

def batchify(batch):
    inputs = []
    targets = []
    for b in batch:
        inputs.append(b[0])
        targets.append(b[1])
    return torch.from_numpy(np.stack(inputs, axis=0)), torch.from_numpy(np.stack(targets, axis=0))

def get_dataloaders(data_path, batch_size=64, max_train_size=1776, max_test_size=350, shuffle=True, k_fold=2):
    cardio = load_cardio_dataset(data_path, train_size=max_train_size, test_size=max_test_size)
    train_data = cardio['train']
    test_data = cardio['test']

    train_eval_dataloaders = []
    train_dataset = Cardio_Dataset(train_data[0], train_data[1])
    total_size = len(train_data[0])
    seg = int(total_size * 1/k_fold)
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        
        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall, valr))
        
        train_set = torch.utils.data.dataset.Subset(train_dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(train_dataset, val_indices)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=batchify)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=batchify)
        train_eval_dataloaders.append( (train_dataloader, val_dataloader) )

    test_dataset = Cardio_Dataset(test_data[0], test_data[1])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=batchify)
    return train_eval_dataloaders, test_dataloader
