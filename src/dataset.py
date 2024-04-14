import json
import os
import pickle

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


def prepare_train_data_loaders(batch_size=64, split_ratio=0.8, u=7, download=False, save_dir='data/training_loaders'):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.Places365(
        root='data/training', split='train-standard',
        download=download, transform=transform, small=True
    )

    # Splitting the dataset indices
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    
    ## Loop over each category and get the indices of the images based on the ratio
    categories_lengths = []
    pbar = tqdm(total=365, position=0, leave=True, desc='Splitting the training dataset')
    for i in range(365):
        pbar.update(1)
        indices_i = [idx for idx in indices if train_dataset.targets[idx] == i]
        categories_lengths.append(len(indices_i))
        split_i = int(np.floor(split_ratio * len(indices_i)))
        np.random.shuffle(indices_i)
        train_unlab_indices, train_lab_indices = indices_i[:split_i], indices_i[split_i:]
        if i == 0:
            train_unlab_indices_all = train_unlab_indices
            train_lab_indices_all = train_lab_indices
        else:
            train_unlab_indices_all = np.concatenate((train_unlab_indices_all, train_unlab_indices))
            train_lab_indices_all = np.concatenate((train_lab_indices_all, train_lab_indices))

    pbar.close()
    assert len(train_unlab_indices_all) + len(train_lab_indices_all) == dataset_size
    assert len(train_unlab_indices_all) == len(set(train_unlab_indices_all))
    assert len(train_lab_indices_all) == len(set(train_lab_indices_all))

    # Creating the dataloaders
    train_lab_dataset = Subset(train_dataset, train_lab_indices_all)
    train_unlab_dataset = Subset(train_dataset, train_unlab_indices_all)

    train_unlab_batch_size = u * batch_size
    train_lab_batch_size = batch_size 

    train_lab_loader = DataLoader(
        train_lab_dataset, batch_size=train_lab_batch_size,
        shuffle=True, num_workers=0,
    )

    train_unlab_loader = DataLoader(
        train_unlab_dataset, batch_size=train_unlab_batch_size,
        shuffle=True, num_workers=0,
    )

    metadata = {
        'train_lab_data_size': len(train_lab_dataset),
        'train_unlab_data_size': len(train_unlab_dataset),
        'categories_lengths': {i: categories_lengths[i] for i in range(365)},
        'arguments': {
            'batch_size': batch_size,
            'split_ratio': split_ratio,
            'u': u,
        },
    }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save train_lab_loader
    with open(os.path.join(save_dir, 'train_lab_loader.pkl'), 'wb') as f:
        pickle.dump(train_lab_loader, f)

    # Save train_unlab_loader
    with open(os.path.join(save_dir,'train_unlab_loader.pkl'), 'wb') as f:
        pickle.dump(train_unlab_loader, f)

    # Save metadata
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)


def prepare_val_and_test_data_loaders(batch_size=64, split_ratio=0.5, download=False, save_dir='data/validation_loaders'):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    val_and_test_dataset = datasets.Places365(
        root='data/validation', split='val',
        download=download, transform=transform,small=True,
    )
    
    dataset_size = len(val_and_test_dataset)
    indices = list(range(dataset_size))
    
    ## Loop over each category and get the indices of the images based on the ratio
    categories_lengths = []
    pbar = tqdm(total=365, position=0, leave=True, desc='Splitting the validation dataset')
    for i in range(365):
        pbar.update(1)
        indices_i = [idx for idx in indices if val_and_test_dataset.targets[idx] == i]
        categories_lengths.append(len(indices_i))
        split_i = int(np.floor(split_ratio * len(indices_i)))
        np.random.shuffle(indices_i)
        val_indices, test_indices = indices_i[:split_i], indices_i[split_i:]
        if i == 0:
            val_indices_all = val_indices
            test_indices_all = test_indices
        else:
            val_indices_all = np.concatenate((val_indices_all, val_indices))
            test_indices_all = np.concatenate((test_indices_all, test_indices))

    pbar.close()
    assert len(val_indices_all) + len(test_indices_all) == dataset_size
    assert len(val_indices_all) == len(set(val_indices_all))
    assert len(test_indices_all) == len(set(test_indices_all))

    # Creating the dataloaders
    val_dataset = Subset(val_and_test_dataset, val_indices_all)
    test_dataset = Subset(val_and_test_dataset, test_indices_all)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4,
    )

    metadata = {
        'val_data_size': len(val_dataset),
        'test_data_size': len(test_dataset),
        'categories_lengths': {i: categories_lengths[i] for i in range(365)},
        'argumets': {
            'batch_size': batch_size,
            'split_ratio': split_ratio,
        },
    }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save val_loader
    with open(os.path.join(save_dir, 'val_loader.pkl'), 'wb') as f:
        pickle.dump(val_loader, f)

    # Save test_loader
    with open(os.path.join(save_dir, 'test_loader.pkl'), 'wb') as f:
        pickle.dump(test_loader, f)

    # Save metadata as json
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    

def get_train_data_loaders(directory='data/training_loaders'):
    with open(os.path.join(directory, 'train_lab_loader.pkl'), 'rb') as f:
        train_lab_loader = pickle.load(f)

    with open(os.path.join(directory, 'train_unlab_loader.pkl'), 'rb') as f:
        train_unlab_loader = pickle.load(f)

    return train_lab_loader, train_unlab_loader

def get_val_and_test_data_loaders(directory='data/validation_loaders'):
    with open(os.path.join(directory, 'val_loader.pkl'), 'rb') as f:
        val_loader = pickle.load(f)

    with open(os.path.join(directory, 'test_loader.pkl'), 'rb') as f:
        test_loader = pickle.load(f)

    return val_loader, test_loader



if __name__ == "__main__":
    prepare_train_data_loaders()
    # train_lab_loader, train_unlab_loader = get_train_data_loaders()
    # print(len(train_lab_loader))
    # print(len(train_unlab_loader))
    # prepare_val_and_test_data_loaders() 
    # val_loader, test_loader = get_val_and_test_data_loaders()
    # print(len(val_loader))
    # print(len(test_loader))