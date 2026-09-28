import numpy as np
import torch
import torch.optim as optim
import sys
from tqdm import trange
import os
from logger import Logger
from test import valid
from loss import MatchLoss
from utils import tocuda
from warmupMultiStepLR import WarmupMultiStepLR


def train_step(step, optimizer, model, match_loss, data, scheduler):
    model.train()
    xs = data['xs']
    ys = data['ys'].squeeze(-1)
    logits, ys_ds, e_hat, y_hat, all_vars = model(xs, ys)
    loss, geo_loss, cla_loss, unc_loss, _, _ = match_loss.run(
        step, data, logits, ys_ds, e_hat, y_hat,
        vars=all_vars
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return [geo_loss, cla_loss, unc_loss]


def train(model, train_loader, valid_loader, config):
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay=config.weight_decay)

    scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=[200000,400000],
        warmup_iters=100000,
        warmup_factor=0.01,
        warmup_method='linear'
    )

    match_loss = MatchLoss(config)

    checkpoint_path = os.path.join(config.log_path, 'checkpoint.pth')
    config.resume = os.path.isfile(checkpoint_path)

    if config.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        best_acc = checkpoint['best_acc']
        start_step = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"Scheduler state loaded from checkpoint (step={start_step})")
        else:
            scheduler.step(start_step)
            print(f"Scheduler not found in checkpoint, manually set to step={start_step}")

        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan', resume=True)
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan', resume=True)

    else:
        best_acc = -1
        start_step = 0
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan')
        logger_train.set_names(
            ['Learning Rate'] +
            ['Geo Loss', 'Classfi Loss', 'Uncertainty Loss'] * (config.iter_num + 1)
        )
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan')
        logger_valid.set_names(['Valid Acc'] + ['Geo Loss', 'Clasfi Loss', 'Uncertainty Loss'])

    train_loader_iter = iter(train_loader)

    for step in trange(start_step, config.train_iter, ncols=config.tqdm_width):
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)
        train_data = tocuda(train_data)

        # run training
        cur_lr = optimizer.param_groups[0]['lr']
        loss_vals = train_step(step, optimizer, model, match_loss, train_data, scheduler)
        logger_train.append([cur_lr] + loss_vals)

        # Check validation
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0

        if b_validate:
            va_res, geo_loss, cla_loss, unc_loss, _, _, _ = valid(valid_loader, model, step, config)
            logger_valid.append([va_res, geo_loss, cla_loss, unc_loss])

            if va_res > best_acc:
                print(f"Saving best model with va_res = {va_res}")
                best_acc = va_res
                torch.save({
                    'epoch': step + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(config.log_path, 'model_best.pth'))
            if cla_loss < 1:
                print(f"Saving best model with cla_loss = {cla_loss}")
                torch.save({
                    'epoch': step + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(config.log_path, f'model_best{cla_loss}.pth'))

        if b_save:
            torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, checkpoint_path)