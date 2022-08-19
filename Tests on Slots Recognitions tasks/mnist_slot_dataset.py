from torchvision import datasets, transforms
from PIL import Image, ImageDraw
import torch
import numpy as np
import random


def load_mnist_datasets(root, normalize=True):
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

    #train_labels = torch.nn.functional.one_hot(train_labels, num_classes=10).float().squeeze(1)
    #test_labels = torch.nn.functional.one_hot(test_labels, num_classes=10).float().squeeze(1)

    mnist_datasets = {
        'train': (train_data, train_labels),
        'test': (test_data, test_labels),
    }
    return mnist_datasets


class Slot_MNIST_Dataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset, dic_class_img_gen, num_examples=10_000, noise=0.):
        super().__init__()
        key_classes = list(dic_class_img_gen.keys())
        generators = {c : dic_class_img_gen[key_classes[c]](mnist_dataset) for c in key_classes}
        num_classes = len(key_classes)
        classes = torch.LongTensor(np.random.choice(key_classes, size=num_examples, replace=True))
        self.classes = torch.nn.functional.one_hot(classes, num_classes=num_classes).float()
        self.images = []
        for c in classes.numpy():
            self.images.append( generators[c]() )
        self.images = np.array(self.images)
        if noise > 0.:
            for i in range(self.images.shape[0]):
                self.images[i] += np.random.random((4, 28, 28))*2*noise - noise
                self.images[i] = np.clip(self.images[i], 0., 1.)
    
    def __len__(self):
        return self.classes.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.images[i])).float(), self.classes[i]
    
    def get_nb_chans(self):
        return 1


class ClassesBucketSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset
        self.len_ = self.dataset.classes.size(-1)
        
    def __iter__(self):
        batches = []
        for c in range(self.len_):
            mapping = torch.where(np.argmax(self.dataset.classes, axis=-1) == c)[0]
            batches.append(mapping)
        return iter(batches)
        
    def __len__(self):
        return self.len_


def collate_fn(x):
    imgs = []
    classes = []
    for b in x:
        img = b[0]
        class_ = b[1]
        imgs.append(img)
        classes.append(class_)
    imgs = torch.stack(imgs, dim=0)
    classes = torch.stack(classes, dim=0)
    return imgs, classes


def get_dataloaders(data_path, dic_class_img_gen, num_train_size, batch_size, noise=0., shuffle=True, num_test_size=0, k_fold=1):
    mnist = load_mnist_datasets(data_path, normalize=False)
    
    train_eval_dataloaders = []
    train_dataset = Slot_MNIST_Dataset(mnist['train'], dic_class_img_gen, num_train_size, noise=noise)
    total_size = len(train_dataset)

    if (k_fold == 0) or (k_fold is None):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate_fn)
        train_eval_dataloaders.append( (train_dataloader, None) )

    elif k_fold == 1:
        indicies = list(range(total_size))
        random.shuffle(indicies)
        train_indices = indicies[:int(2./3. * total_size)]
        val_indices = indicies[int(2./3. * total_size):]
        train_set = torch.utils.data.dataset.Subset(train_dataset, train_indices)
        val_set = torch.utils.data.dataset.Subset(train_dataset, val_indices)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate_fn)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate_fn)
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
            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate_fn)
            val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=collate_fn)
            train_eval_dataloaders.append( (train_dataloader, val_dataloader) )
    
    test_dataset = Slot_MNIST_Dataset(mnist['test'], dic_class_img_gen, num_test_size, noise=noise)
    batchSampler = ClassesBucketSampler(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=batchSampler,
                                                  num_workers=2, collate_fn=collate_fn)
    return train_eval_dataloaders, test_dataloader


## DIGIT GENERATION
def generate_empty_img():
    image = Image.new('1', (28, 28))
    draw = ImageDraw.Draw(image)
    return np.array(image.getdata(), dtype=float).reshape((28, 28))/255.

class Generate_MNIST_digit():
    def __init__(self, digits, mnist_data):
        self.digits = digits
        self.mnist_data = mnist_data
        self.dic_idx = {digit: torch.where(mnist_data[1] == digit)[0] for digit in set(self.digits) if digit is not None}
        self.dic_idx_size = {digit: self.dic_idx[digit].size(0) for digit in set(self.digits) if digit is not None}
        
    def __call__(self):
        list_idx = [-1 if digit is None else torch.randint(low=0, high=self.dic_idx_size[digit], size=(1,)).item() for digit in self.digits]
        return np.array([generate_empty_img() if digit is None else self.mnist_data[0][self.dic_idx[digit][idx]].squeeze(0).numpy() for digit, idx in zip(self.digits, list_idx)])



