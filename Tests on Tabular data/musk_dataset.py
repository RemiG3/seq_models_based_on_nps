import numpy as np
import random
import torch

def load_musk_dataset(data_path, train_size=5500, test_size=1000):
    with open(data_path, 'r') as f:
        list_data = [[int(e) if(e.isdigit() or (e[1:].isdigit() and e[0] == '-')) else float(e) if(e[:-1].isdigit() and e[-1] == '.') else 0 for e in line.replace('\n', '').split(',')] for line in f.readlines()]
    list_data = np.array(list_data)
    np.random.shuffle(list_data)
    list_data_features = list_data[:,2:-1]
    list_data_cat = list_data[:,-1]
    
    labels = torch.LongTensor(list_data_cat).unsqueeze(1)
    labels = torch.nn.functional.one_hot(labels, num_classes=len(set(list_data_cat))).float().squeeze(1)
    data = torch.FloatTensor(list_data_features)
    
    train_labels, train_data = labels[:train_size], data[:train_size]
    test_labels, test_data = labels[train_size:train_size+test_size], data[train_size:train_size+test_size]
    
    letter_dataset = {
        'train': (train_data, train_labels),
        'test': (test_data, test_labels),
    }
    
    assert train_size+test_size <= list_data.shape[0]
    return letter_dataset

class Musk_Dataset(torch.utils.data.Dataset):
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


def get_dataloaders(data_path, batch_size=64, max_train_size=5500, max_test_size=1000, shuffle=True, k_fold=2): # data_path='./Musk/V2/clean2.data'
    letters = load_musk_dataset(data_path, train_size=max_train_size, test_size=max_test_size)
    train_data = letters['train']
    test_data = letters['test']

    train_eval_dataloaders = []
    train_dataset = Musk_Dataset(train_data[0], train_data[1])
    total_size = len(train_data[0])

    if (k_fold == 0) or (k_fold is None):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=batchify)
        train_eval_dataloaders.append( (train_dataloader, None) )

    elif k_fold == 1:
        indicies = list(range(total_size))
        random.shuffle(indicies)
        train_indices = indicies[:int(2./3. * total_size)]
        val_indices = indicies[int(2./3. * total_size):]
        train_set = torch.utils.data.dataset.Subset(train_dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(train_dataset, val_indices)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=batchify)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=batchify)
        train_eval_dataloaders.append( (train_dataloader, val_dataloader) )

    else: # k_fold > 1
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

    test_dataset = Musk_Dataset(test_data[0], test_data[1])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=batchify)
    return train_eval_dataloaders, test_dataloader
