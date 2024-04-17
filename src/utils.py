import random

import torch
import torchvision.transforms.functional as TF


# Define a function that weakly augments the images
def weak_augment(images):
    images = TF.hflip(images) if random.random() > 0.5 else images
    images = TF.adjust_brightness(images, brightness_factor=random.uniform(0.8, 1.2))
    images = TF.adjust_contrast(images, contrast_factor=random.uniform(0.2, 1.8))
    return images

# Define a function that very strongly augments the images
def strong_augment(images):
    images = TF.hflip(images) if random.random() > 0.5 else images
    images = TF.adjust_brightness(images, brightness_factor=random.uniform(0.2, 1.8))
    images = TF.adjust_contrast(images, contrast_factor=random.uniform(0.2, 1.8))
    images = TF.adjust_saturation(images, saturation_factor=random.uniform(0.2, 1.8))
    images = TF.adjust_hue(images, hue_factor=random.uniform(-0.2, 0.2))
    return images

# Define a function that creates pseudo labels above a threshold of confidence and assign -1 to the others
def create_pseudo_labels(model, images, threshold):
    # model(images) should return a tensor of shape (batch_size, num_classes)
    # with logits for each class
    outputs = model(images)
    probabilities = torch.softmax(outputs, dim=1) # Convert logits to probabilities
    max_probs, max_indexes = torch.max(probabilities, dim=1)
    # print(f"Max probsw: {max_probs}")
    
    pseudo_labels = torch.where(max_probs > threshold, max_indexes, torch.tensor(-1).to(max_indexes.device))
    return pseudo_labels

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum(1)
        accuracy_value = correct_k.sum().float() * (100/ batch_size)
        res.append(accuracy_value)
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count