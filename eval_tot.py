import time
import torch

from timm.utils import accuracy, AverageMeter
from utils import reduce_tensor, get_coarse_targets, DS_Combin


@torch.no_grad()
def validate(config, tot, data_loader, model, logger):
    model.eval()

    acc_c_meter = [AverageMeter() for _ in range(tot.n_layer)]
    acc_meter = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        c_targets = get_coarse_targets(target, tot)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        acc_c = []
        for i in range(tot.n_layer):
            target_c = c_targets[i+1] - tot.coarse_cache["label2tid"][i].min() - 1
            pred_c = output[0][i].max(dim=-1)[1]
            acc_c.append(pred_c.eq(target_c).to(torch.float32).mean(dim=0))

        pred = output[1].max(dim=-1)[1]
        acc_l = pred.eq(target).to(torch.float32).mean()

        if config.DIST:
            for i in range(tot.n_layer):
                acc_c[i] = reduce_tensor(acc_c[i])
            acc_l = reduce_tensor(acc_l)

        for i in range(tot.n_layer):
            acc_c_meter[i].update(acc_c[i], target.size(0))
        acc_meter.update(acc_l, target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Coarse-1 Acc@1 {acc_c_meter[0].val.mean().item():.4f} ({acc_c_meter[0].avg.mean().item():.4f})\t'
                f'Coarse-2 Acc@1 {acc_c_meter[1].val.mean().item():.4f} ({acc_c_meter[1].avg.mean().item():.4f})\t'
                f'Coarse-3 Acc@1 {acc_c_meter[2].val.mean().item():.4f} ({acc_c_meter[2].avg.mean().item():.4f})\t'
                f'===============\t'
                f'leaves Acc@1 {acc_meter.val:.4f} ({acc_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')
    acc_c = [[float('{:.4f}'.format(meter)) for meter in acc_c_meter[i].avg.tolist()] for i in range(tot.n_layer)]
    acc_c_t = [acc_c_meter[i].avg.mean().item() for i in range(tot.n_layer)]
    logger.info(
        f'\n'
        f' * Coarse-1 Acc@1 {acc_c[0]}\n'
        f' * Coarse-2 Acc@1 {acc_c[1]}\n'
        f' * Coarse-3 Acc@1 {acc_c[2]}\n'
        f' * leaves Acc@1 {acc_meter.avg:.4f}\n')
    return acc_c_t, acc_meter.avg.item()
