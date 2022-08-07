from torchvision import datasets, transforms
import torch
import numpy as np



def load_cifar10_datasets(root, extrap=False, max_train_size=None, max_eval_size=None):
    train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
    valtest_dataset = datasets.CIFAR10(root, train=False, transform=transforms.Compose([transforms.ToTensor(),]))
    
    train_data, train_labels = zip(*train_dataset)
    train_data = torch.stack(train_data)
    train_labels = torch.LongTensor(train_labels).unsqueeze(1)
    valtest_data, valtest_labels = zip(*valtest_dataset)
    valtest_data = torch.stack(valtest_data)
    valtest_labels = torch.LongTensor(valtest_labels).unsqueeze(1)

    if max_train_size is not None:
        train_data, train_labels = train_data[:max_train_size], train_labels[:max_train_size]
    if max_eval_size is not None:
        valtest_data, valtest_labels = valtest_data[:max_eval_size], valtest_labels[:max_eval_size]
    
    num_classes = len(set(train_labels.reshape((-1,)).numpy()))
    train_labels = torch.nn.functional.one_hot(train_labels, num_classes=num_classes).float().squeeze(1)
    valtest_labels = torch.nn.functional.one_hot(valtest_labels, num_classes=num_classes).float().squeeze(1)

    numtest = int(len(valtest_data) / 2)
    val_data = valtest_data[:numtest]
    val_labels = valtest_labels[:numtest]
    test_data = valtest_data[numtest:]
    test_labels = valtest_labels[numtest:]

    cifar10_dataset = {
        'train': (train_data, train_labels),
        'val': (val_data, val_labels),
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


def get_dataloaders(data_path, batch_size=64, max_train_size=None, max_eval_size=None, shuffle=True):
    cifar10 = load_cifar10_datasets(data_path, max_train_size=max_train_size, max_eval_size=max_eval_size)
    train_data = cifar10['train']
    val_data = cifar10['val']
    test_data = cifar10['test']
    train_dataset = CIFAR10_Dataset(train_data[0], train_data[1])
    val_dataset = CIFAR10_Dataset(val_data[0], val_data[1])
    test_dataset = CIFAR10_Dataset(test_data[0], test_data[1])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=batchify)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=batchify)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=batchify)

    return train_dataloader, val_dataloader, test_dataloader