import torch

from train_sparse_mask import loss_calc
from sparse_mask_train_mode import Decoder


def sparse_loss(w, ratio=0.01, eps=1e-10):
    w = w / 2
    m = torch.mean(w)

    return - torch.mean(w*torch.log(w+eps) + (1-w)*torch.log(1-w+eps)) - (
            ratio*torch.log(m+eps) + (1-ratio)*torch.log(1-m+eps))


def run(model, criterion, optimizer, data, target, scheduler, iter_idx, args):
    output = model(data)

    loss_task = loss_calc(criterion, output, target)
    # Sparse
    loss_sparse = sum([sparse_loss(m.weight, ratio=min(2.0/m.weight.numel(), 0.5))
                       for m in model.modules() if isinstance(m, Decoder)])
    loss = loss_task + args.sparse*loss_sparse
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    for m in model.modules():
        if isinstance(m, Decoder):
            m._weight_.data.clamp_(0, 2)

    # log
    if iter_idx % args.log_interval == 0:
        args.writer.add_scalar('train/backbone_lr', optimizer.param_groups[0]['lr'], iter_idx)
        args.writer.add_scalar('train/decoder_lr', optimizer.param_groups[1]['lr'], iter_idx)
        args.writer.add_scalar('train/loss', loss.item(), iter_idx)
        args.writer.add_scalar('train/loss_task', loss_task.item(), iter_idx)
        args.writer.add_scalar('train/loss_sparse', loss_sparse.item(), iter_idx)

        for name, param in model.named_parameters():
            args.writer.add_histogram(name, param.data.cpu().numpy(), iter_idx)
