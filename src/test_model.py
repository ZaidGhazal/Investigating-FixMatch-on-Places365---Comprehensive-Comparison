

import time

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

from dataset import get_val_and_test_data_loaders
from src.utils import AverageMeter, accuracy


def test_model(model_path, arch, device, criterion, path_to_logs):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # Load the model
    checkpoint = torch.load(model_path)
    modified_state_dict = {
        k.replace(".module", ""): v for k, v in checkpoint['state_dict'].items()
    }
    model = models.__dict__[arch]()
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 365)
    model.load_state_dict(modified_state_dict)
    model.eval()
    model.to(device)
    
    # Load the test data
    _, test_loader = get_val_and_test_data_loaders()
    results_log = pd.DataFrame()
        
    end = time.time()
    # Evaluate the model
    for i, (input, target) in enumerate(test_loader):
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
                'step': f"{i}/{len(test_loader)}",
                'test_loss': losses.val,
                'test_acc_top1': top1.val,
                'test_acc_top5': top5.val,
                'avg_test_acc_top1': top1.avg,
                'avg_test_acc_top5': top5.avg,
                'avg_test_loss': losses.avg,
            }, index=[0])], ignore_index=True)

            if i % 5 == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))
    
    results_log.to_csv(f'{path_to_logs}/test_results.csv', index=False)




test_model('alexnet_best.pth.tar', 'alexnet', 'mps', nn.CrossEntropyLoss(),  '/Users/zghazal/Desktop/online-repos/Investigating-FixMatch-on-Places365-Comprehensive-Comparison/src/logs/run_at_2024-04-16 21:23:11')