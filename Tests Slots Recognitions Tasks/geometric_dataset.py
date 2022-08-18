import torch
import random
import numpy as np

from PIL import Image, ImageDraw



## SHAPE GENERATION (random placement)
def get_rand_val(min_range, max_range):
    return int(round(min_range + random.random() * (max_range-min_range)))

def get_rand_center_radius(range_center, range_radius):
    if isinstance(range_center[0], (tuple, list)):
        center = (get_rand_val(*range_center[0]), get_rand_val(*range_center[1]))
    else:
        center = (get_rand_val(*range_center), get_rand_val(*range_center))
    radius = get_rand_val(*range_radius)
    return center, radius

def generate_circle(draw, range_center, range_radius, color='black', fill='white'):
    center, radius = get_rand_center_radius(range_center, range_radius)
    draw.ellipse((center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius), outline=color, fill=fill)

def generate_triangle(draw, range_center, range_radius, color='black', fill='white'):
    center, radius = get_rand_center_radius(range_center, range_radius)
    draw.polygon([(center[0], center[1]-radius), (center[0]+radius, center[1]), (center[0]-radius, center[1])], outline=color, fill=fill)

def generate_diamond(draw, range_center, range_radius, color='black', fill='white'):
    center, radius = get_rand_center_radius(range_center, range_radius)
    draw.polygon([(center[0], center[1]+radius), (center[0]-radius, center[1]), (center[0], center[1]-radius), (center[0]+radius, center[1])], outline=color, fill=fill)

def generate_square(draw, range_center, range_radius, color='black', fill='white'):
    center, radius = get_rand_center_radius(range_center, range_radius)
    draw.polygon([(center[0]+radius, center[1]+radius), (center[0]+radius, center[1]-radius), (center[0]-radius, center[1]-radius), (center[0]-radius, center[1]+radius)], outline=color, fill=fill)


## IMAGE GENERATION
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])/255

def generate_img(callback_generator, **kwargs):
    image = Image.new('1', (10, 10))
    draw = ImageDraw.Draw(image)
    if callback_generator is not None:
        callback_generator(draw, **kwargs)
    return np.array(image.getdata()).reshape(10,10)/255.


class Geometric_Dataset(torch.utils.data.Dataset):
    def __init__(self, dic_class_img_gen, num_examples=10_000, noise=0.):
        super().__init__()
        key_classes = list(dic_class_img_gen.keys())
        num_classes = len(key_classes)
        classes = torch.LongTensor(np.random.choice(key_classes, size=num_examples, replace=True))
        self.classes = torch.nn.functional.one_hot(classes, num_classes=num_classes).float()
        self.images = []
        for c in classes:
            self.images.append( dic_class_img_gen[key_classes[c]]() )
        self.images = np.array(self.images)
        if noise > 0.:
            for i in range(self.images.shape[0]):
                self.images[i] += np.random.random((4, 10, 10))*2*noise - noise
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


def get_dataloaders(dic_class_img_gen, num_train_size, batch_size, noise=0., num_test_size=0, shuffle=True, k_fold=1):
    train_eval_dataloaders = []
    train_dataset = train_dataset = Geometric_Dataset(dic_class_img_gen, num_train_size, noise=noise)
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
    
    test_dataset = Geometric_Dataset(dic_class_img_gen, num_test_size, noise=noise)
    batchSampler = ClassesBucketSampler(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=batchSampler,
                                                  num_workers=2, collate_fn=collate_fn)
    return train_eval_dataloaders, test_dataloader
