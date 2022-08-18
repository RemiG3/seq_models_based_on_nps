from torchvision import datasets, transforms
import torch
import numpy as np
import random


def load_cifar10_datasets(root, max_train_size=None, max_test_size=None):
    train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
    test_dataset = datasets.CIFAR10(root, train=False, transform=transforms.Compose([transforms.ToTensor(),]))
    
    train_data, train_labels = zip(*train_dataset)
    train_data = torch.stack(train_data)
    train_labels = torch.LongTensor(train_labels).unsqueeze(1)
    test_data, test_labels = zip(*test_dataset)
    test_data = torch.stack(test_data)
    test_labels = torch.LongTensor(test_labels).unsqueeze(1)

    if max_train_size is not None:
        train_data, train_labels = train_data[:max_train_size], train_labels[:max_train_size]
    if max_test_size is not None:
        test_data, test_labels = test_data[:max_test_size], test_labels[:max_test_size]
    
    num_classes = len(set(train_labels.reshape((-1,)).numpy()))
    train_labels = torch.nn.functional.one_hot(train_labels, num_classes=num_classes).float().squeeze(1)
    test_labels = torch.nn.functional.one_hot(test_labels, num_classes=num_classes).float().squeeze(1)

    cifar10_dataset = {
        'train': (train_data, train_labels),
        'test': (test_data, test_labels),
    }

    return cifar10_dataset


class CIFAR10_Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        """     
            inputs: (bsize, 1, H, W)
            targets: (bsize, 1)
        """
        super().__init__()
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        inp = self.inputs[i]
        tar = self.targets[i]
        return inp, tar
    
    def get_nb_chans(self):
        return self.inputs.shape[1]


def batchify(batch):
    inputs = []
    targets = []
    for b in batch:
        inputs.append(b[0])
        targets.append(b[1])
    return torch.from_numpy(np.stack(inputs, axis=0)), torch.from_numpy(np.stack(targets, axis=0))


def get_dataloaders(data_path, batch_size=64, max_train_size=None, max_test_size=None, shuffle=True, k_fold=1):
    cifar10 = load_cifar10_datasets(data_path, max_train_size=max_train_size, max_test_size=max_test_size)
    train_data = cifar10['train']
    test_data = cifar10['test']

    train_eval_dataloaders = []
    train_dataset = CIFAR10_Dataset(train_data[0], train_data[1])
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

    test_dataset = CIFAR10_Dataset(test_data[0], test_data[1])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=batchify)
    return train_eval_dataloaders, test_dataloader
