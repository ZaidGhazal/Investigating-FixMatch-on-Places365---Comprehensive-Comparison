import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm

import wideresnet
from dataset import get_train_data_loaders, get_val_and_test_data_loaders
from src import train
from src.validation import validate

desired_models = ["alexnet", "vgg16", "wideresnet", "googlenet"]


def main(start_epoch, epochs, arch, nesterov=True, lr=0.03, momentum=0.9, weight_decay=0.0005, batch_size=64, num_classes=365, resume: str = None, device: str = 'mps'):
    device = torch.device(device) if device == 'mps' else device
    
    print("=> creating model '{}'".format(arch))
   
    if arch.lower().startswith('wideresnet'):
        model  = wideresnet.resnet152(pretrained=True, num_classes=num_classes)
    else:
        model = models.__dict__[arch](num_classes=num_classes)

    if arch.lower().startswith('alexnet') or arch.lower().startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.to(device)
    else:
        model = torch.nn.DataParallel(model).to(device)
    print (model)
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay,
                                nesterov=nesterov)

    train_lab_loader, train_unlab_loader = get_train_data_loaders()
    val_loader, test_loader = get_val_and_test_data_loaders()
    # pbar = tqdm(range(epochs), position=0, leave=True, desc="Training")

    best_prec1 = 0
    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        train.train(
            train_lab_loader=train_lab_loader,
            train_unlab_loader=train_unlab_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            lambda_u=1.0,
            tau=0.95,
        )

        # # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, device=device)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, arch.lower())

        # pbar.update(1)
    # pbar.close()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main(start_epoch=0, 
         epochs=20, 
         arch='alexnet', 
         lr=0.03, 
         momentum=0.9, 
         weight_decay=0.0005, 
         batch_size=64, 
         num_classes=365, 
         resume=None, 
         device='mps')