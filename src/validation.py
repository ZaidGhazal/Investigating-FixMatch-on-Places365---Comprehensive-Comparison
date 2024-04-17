import time

import pandas as pd
import torch

from src.utils import AverageMeter, accuracy


def validate(epoch, val_loader, model, criterion, device, results_log, path_to_logs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input = input.to(device)
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            results_log = pd.concat([results_log, pd.DataFrame({
                'epoch': epoch,
                'step': f"{i}/{len(val_loader)}",
                'val_loss': losses.val,
                'val_acc_top1': top1.val,
                'val_acc_top5': top5.val,
                'avg_val_loss': losses.avg,
                'avg_val_acc_top1': top1.avg,
                'avg_val_acc_top5': top5.avg
            }, index=[0])], ignore_index=True)

            if i % 5 == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
        
        results_log.to_csv(f'{path_to_logs}/validation_results.csv', index=False)

        return top5.avg