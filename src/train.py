import random
import time

import pandas as pd
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
    used_pseudo_labels_count = filtered_pseudo_labels.shape[0]
    if filtered_pseudo_labels.shape[0] > 0:
        print(f"Filtered pseudo labels shape: {filtered_pseudo_labels.shape}")
        loss_value_unlab = loss_fn(filtered_y_pred_sa, filtered_pseudo_labels)
    else:
        loss_value_unlab = torch.tensor(0)

    # Compute the total loss
    loss_value = loss_value_lab + lambda_u * loss_value_unlab

    return loss_value, y_pred_lab, used_pseudo_labels_count

            

def train(train_lab_loader, train_unlab_loader,  model, criterion, optimizer, epoch, device, lambda_u, tau, results_log, path_to_logs, only_lab=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    if only_lab:
        min_len = len(train_lab_loader)
    else:
        min_len = min(len(train_lab_loader), len(train_unlab_loader))
    # pbar = tqdm(min_len, position=0, leave=True, desc='Training - Epoch {}'.format(epoch))
    for i in range(min_len):
        start = time.time()
        input_lab, target_lab = next(iter(train_lab_loader))
        input_lab = input_lab.to(device)
        target_lab = target_lab.to(device)
        input_lab_var = torch.autograd.Variable(input_lab)
        target_lab_var = torch.autograd.Variable(target_lab)
        # print(f"----Labeled batch reading time: {time.time() - start}----")
        
        start = time.time()
        if not only_lab:
            input_unlab, _ = next(iter(train_unlab_loader))
            input_unlab = input_unlab.to(device)
            input_unlab_var = torch.autograd.Variable(input_unlab)
        # print(f"----Unlabeled batch reading time: {time.time() - start}----")
         
        data_time.update(time.time() - end)

        

        # Compute total loss
        start = time.time()
        if not only_lab:
            loss_value, y_pred_lab, used_pseudo_labels_count = compute_total_loss(
                model=model, x_batch_lab=input_lab_var, y_batch_lab=target_lab_var, loss_fn=criterion,
                x_batch_unlab=input_unlab_var, tau=tau, lambda_u=lambda_u)
        else:
            used_pseudo_labels_count = 0
            y_pred_lab = model(input_lab_var)
            loss_value = criterion(y_pred_lab, target_lab_var)
        # print(f"---- Compute total loss time: {time.time() - start} -----")

        total_inputs_size = input_lab.size(0) + used_pseudo_labels_count
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


        if i % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, min_len, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
        if i % 10 == 0:
            results_log = pd.concat([results_log, pd.DataFrame({
                'epoch': epoch,
                'step': f"{i}/{min_len}",
                'train_loss': losses.val,
                'train_acc_top1': top1.val,
                'train_acc_top5': top5.val,
                'avg_train_loss': losses.avg,
                'avg_train_acc_top1': top1.avg,
                'avg_train_acc_top5': top5.avg,
            }, index=[0])], ignore_index=True)
            results_log.to_csv(f'{path_to_logs}/training_results.csv', index=False)
    #     pbar.update(1)
    # pbar.close()
            