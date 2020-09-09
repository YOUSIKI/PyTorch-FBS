# -*- coding=utf-8 -*-

__all__ = [
    'AverageMeter',
    'AccuracyMeter'
]


import torch


class AverageMeter(object):
    """
    Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L359
    """

    def __init__(self, name):
        self.name = name
        self.reset()

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg}"


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L407
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res


class AccuracyMeter(object):

    def __init__(self, topk=(1,)):
        self.topk = topk
        self.meters = [AverageMeter("Acc@%d" % i) for i in topk]

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def reset(self):
        for meter in self.meters:
            meter.reset()

    def update(self, predict, target):
        acc = accuracy(predict, target, self.topk)
        num = target.size(0)
        for meter, val in zip(self.meters, acc):
            meter.update(val, num)
        return self.get()

    def __str__(self):
        return " ".join(str(meter) for meter in self.meters)

    def get(self):
        return {meter.name: meter.avg for meter in self.meters}
    
    def top(self):
        return self.meters[0].avg
