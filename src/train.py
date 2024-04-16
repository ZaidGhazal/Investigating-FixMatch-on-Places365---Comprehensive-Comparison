import random
import time

import torch
from tqdm import tqdm

from src.utils import (AverageMeter, accuracy, create_pseudo_labels,
                       strong_augment, weak_augment)


# Define a function to compute the total loss
def compute_total_loss(model, x_batch_lab, y_batch_lab, loss_fn, x_batch_unlab, tau, lambda_u):
    # Get predictions for the lab batch
    start = time.time()
    y_pred_lab = model(x_batch_lab)
    # print(f"Model prediction on labeled data time: {time.time() - start}")

    start = time.time()
    loss_value_lab = loss_fn(y_pred_lab, y_batch_lab)
    # print(f"Loss computation labeled data time: {time.time() - start}")

    # Get predictions for the unlab batch (weakly augmented)
    start = time.time()
    x_batch_unlab_wa = weak_augment(x_batch_unlab)
    # print(f"Weak augmentation time: {time.time() - start}")

    # Get predictions for the unlab batch (strongly augmented)
    start = time.time()
    x_batch_unlab_sa = strong_augment(x_batch_unlab)
    # print(f"Strong augmentation time: {time.time() - start}")

    start = time.time()
    y_pred_sa = model(x_batch_unlab_sa)
    # print(f"Model prediction on strongly augmented data time: {time.time() - start}")

    # Compute the loss for valid pseudo labels
    start = time.time()
    pseudo_labels = create_pseudo_labels(model, x_batch_unlab_wa, tau)
    filtered_pseudo_labels = pseudo_labels[pseudo_labels != -1]
    filtered_y_pred_sa = y_pred_sa[pseudo_labels != -1]
    filtered_x_unlab = x_batch_unlab[pseudo_labels != -1]
    # print(f"Create pseudo labels time: {time.time() - start}")

    # Compute the loss for the unlab batch if there are valid pseudo labels
    if filtered_pseudo_labels.shape[0] > 0:
        loss_value_unlab = loss_fn(filtered_pseudo_labels, filtered_y_pred_sa)
    else:
        loss_value_unlab = torch.tensor(0)

    # Compute the total loss
    loss_value = loss_value_lab + lambda_u * loss_value_unlab

    return loss_value, y_pred_lab

            

def train(train_lab_loader, train_unlab_loader,  model, criterion, optimizer, epoch, device, lambda_u, tau):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    min_len = min(len(train_lab_loader), len(train_unlab_loader))
    # pbar = tqdm(min_len, position=0, leave=True, desc='Training - Epoch {}'.format(epoch))
    for i in range(min_len):
        start = time.time()
        input_lab, target_lab = next(iter(train_lab_loader))
        # print(f"----Labeled batch reading time: {time.time() - start}----")
        start = time.time()
        input_unlab, _ = next(iter(train_unlab_loader))
        # print(f"----Unlabeled batch reading time: {time.time() - start}----")
         
        data_time.update(time.time() - end)

        start = time.time()
        input_lab = input_lab.to(device)
        target_lab = target_lab.to(device)
        input_unlab = input_unlab.to(device)
        # print(f"----Data transfer to device time: {time.time() - start}----")
        
        # start = time.time()
        input_lab_var = torch.autograd.Variable(input_lab)
        target_lab_var = torch.autograd.Variable(target_lab)
        input_unlab_var = torch.autograd.Variable(input_unlab)
        # print(f"----Variable creation time: {time.time() - start}----")

        # Compute total loss
        start = time.time()
        loss_value, y_pred_lab = compute_total_loss(
            model=model, x_batch_lab=input_lab_var, y_batch_lab=target_lab_var, loss_fn=criterion,
            x_batch_unlab=input_unlab_var, tau=tau, lambda_u=lambda_u)
        # y_pred_lab = model(input_lab_var)
        # loss_value = criterion(y_pred_lab, target_lab_var)
        # print(f"---- Compute total loss time: {time.time() - start} -----")

        total_inputs_size = input_lab.size(0) + input_unlab.size(0)
        input_lab_size = input_lab.size(0)


        # measure accuracy and record loss - Accuracy is computed only for the lab batch
        start = time.time()
        prec1, prec5 = accuracy(y_pred_lab, target_lab, topk=(1, 5))
        losses.update(loss_value.item(), total_inputs_size)
        top1.update(prec1.item(), input_lab_size)
        top5.update(prec5.item(), input_lab_size)
        # print(f"---- Accuracy computation time: {time.time() - start} -----")

        # compute gradient and do SGD step
        start = time.time()
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        # print(f"---- Gradient computation time: {time.time() - start} -----")

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 2 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, min_len, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
    #     pbar.update(1)
    # pbar.close()
            