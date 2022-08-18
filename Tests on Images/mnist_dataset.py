from torchvision import datasets, transforms
import torch
import numpy as np
import random



def load_mnist_datasets(root, nums_keep=list(range(10)), normalize=True, max_train_size=None, max_test_size=None):
    assert (max_train_size is None) or (max_train_size <= 60_000)
    assert (max_test_size is None) or (max_test_size <= 10_000)

    nums_keep = None if(nums_keep == []) else nums_keep
    if normalize:
        train_dataset = datasets.MNIST(root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        test_dataset = datasets.MNIST(root, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    else:
        train_dataset = datasets.MNIST(root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))
        test_dataset = datasets.MNIST(root, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]))

    train_data, train_labels = zip(*train_dataset)
    train_data = torch.stack(train_data)
    train_labels = torch.LongTensor(train_labels).unsqueeze(1)
    test_data, test_labels = zip(*test_dataset)
    test_data = torch.stack(test_data)
    test_labels = torch.LongTensor(test_labels).unsqueeze(1)

    if nums_keep is not None:
        train_data_idx = torch.LongTensor([])
        test_data_idx = torch.LongTensor([])
        for num_keep in nums_keep:
            train_data_idx = torch.cat((train_data_idx, (train_labels.squeeze(-1) == num_keep).nonzero(as_tuple=False)), 0)
            test_data_idx = torch.cat((test_data_idx, (test_labels.squeeze(-1) == num_keep).nonzero(as_tuple=False)), 0)
        train_data_idx = train_data_idx.sort(dim=0)[0].squeeze(-1)
        test_data_idx = test_data_idx.sort(dim=0)[0].squeeze(-1)
        train_data = train_data[train_data_idx]
        train_labels = train_labels[train_data_idx]
        test_data = test_data[test_data_idx]
        test_labels = test_labels[test_data_idx]

    if max_train_size is not None:
        train_data, train_labels = train_data[:max_train_size], train_labels[:max_train_size]
    if max_test_size is not None:
        test_data, test_labels = test_data[:max_test_size], test_labels[:max_test_size]
    n = 0
    for num in nums_keep:
        train_labels = torch.where((train_labels == num), n, train_labels)
        test_labels = torch.where((test_labels == num), n, test_labels)
        n += 1
    train_labels = torch.nn.functional.one_hot(train_labels, num_classes=len(nums_keep)).float().squeeze(1)
    test_labels = torch.nn.functional.one_hot(test_labels, num_classes=len(nums_keep)).float().squeeze(1)

    mnist_datasets = {
        'train': (train_data, train_labels),
        'test': (test_data, test_labels),
    }
    return mnist_datasets


class MNIST_Dataset(torch.utils.data.Dataset):
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


def get_dataloaders(data_path, nums_keep=list(range(10)), batch_size=64, max_train_size=None, max_test_size=None, shuffle=True, k_fold=1):
    mnist = load_mnist_datasets(data_path, nums_keep=nums_keep, normalize=False, max_train_size=max_train_size, max_test_size=max_test_size)
    train_data = mnist['train']
    test_data = mnist['test']

    train_eval_dataloaders = []
    train_dataset = MNIST_Dataset(train_data[0], train_data[1])
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

    test_dataset = MNIST_Dataset(test_data[0], test_data[1])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=batchify)
    return train_eval_dataloaders, test_dataloader
