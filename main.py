# -*- coding=utf-8 -*-

import os
import torch
from tqdm import tqdm
from loguru import logger
from models import *
from utils import *


def watch_nan(x: torch.Tensor):
    if torch.isnan(x).any():
        raise ValueError('found NaN: ' + str(x))


if __name__ == '__main__':
    name = 'resnet34-imagenette2'
    os.makedirs('log', exist_ok=True)
    os.makedirs('ckpts', exist_ok=True)
    log_path = os.path.join('log', name + '.log')
    if os.path.isfile(log_path):
        os.remove(log_path)
    logger.add(log_path)
    net = resnet34(num_classes=10).cuda()
    train_dl = imagenette2('train')
    valid_dl = imagenette2('val')
    for pruning_rate in [1.0, 0.8, 0.6, 0.4]:
        logger.info('set pruning rate = %.1f' % pruning_rate)
        # set pruning rate
        for m in net.modules():
            if hasattr(m, 'rate'):
                m.rate = pruning_rate
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80], 0.2)
        loss_function = torch.nn.CrossEntropyLoss()
        best_accuracy = 0
        for epoch in range(100):
            with tqdm(train_dl) as train_tqdm:
                train_tqdm.set_description_str('{:03d} train'.format(epoch))
                net.train()
                meter = AccuracyMeter(topk=(1, 5))
                for images, labels in train_tqdm:
                    images = images.cuda()
                    labels = labels.cuda()
                    output = net(images)
                    watch_nan(output)
                    meter.update(output, labels)
                    train_tqdm.set_postfix(meter.get())
                    optimizer.zero_grad()
                    loss1 = loss_function(output, labels)
                    loss2 = 0
                    for m in net.modules():
                        if hasattr(m, 'loss') and m.loss is not None:
                            loss2 += m.loss
                    loss = loss1 + 1e-8 * loss2
                    loss.backward()
                    optimizer.step()
                logger.info('{:03d} train result: {}'.format(epoch, meter.get()))
            with tqdm(valid_dl) as valid_tqdm:
                valid_tqdm.set_description_str('{:03d} valid'.format(epoch))
                net.eval()
                meter = AccuracyMeter(topk=(1, 5))
                with torch.no_grad():
                    for images, labels in valid_tqdm:
                        images = images.cuda()
                        labels = labels.cuda()
                        output = net(images)
                        watch_nan(output)
                        meter.update(output, labels)
                        valid_tqdm.set_postfix(meter.get())
                logger.info('{:03d} valid result: {}'.format(epoch, meter.get()))
            if (epoch + 1) % 10 == 0:
                torch.save(net.state_dict(), 'ckpts/%.1f-latest.pth' % pruning_rate)
                logger.info('saved to ckpts/%.1f-latest.pth' % pruning_rate)
            if best_accuracy < meter.top():
                best_accuracy = meter.top()
                torch.save(net.state_dict(), 'ckpts/%.1f-best.pth' % pruning_rate)
                logger.info('saved to ckpts/%.1f-best.pth' % pruning_rate)
            scheduler.step()
