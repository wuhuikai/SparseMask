from train_sparse_mask import loss_calc


def run(model, criterion, optimizer, data, target, scheduler, iter_idx, args):
    output = model(data)

    loss = loss_calc(criterion, output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # log
    if iter_idx % args.log_interval == 0:
        args.writer.add_scalar('train/base_lr', optimizer.param_groups[0]['lr'], iter_idx)
        args.writer.add_scalar('train/last_layer_lr', optimizer.param_groups[1]['lr'], iter_idx)
        args.writer.add_scalar('train/loss', loss.item(), iter_idx)

        for name, param in model.named_parameters():
            args.writer.add_histogram(name, param.data.cpu().numpy(), iter_idx)
