from __future__ import absolute_import


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class GroupAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}
    def add_key(self, key):
        self.val[key] = 0
        self.avg[key] = 0
        self.sum[key] = 0
        self.count[key] = 0
    def update(self, dic):
        for key,v in dic.items():
            if key not in self.val:
                self.add_key(key)
            value, count = v
            if count==0:
                continue
            self.sum[key] += value*count
            self.count[key] += count
            self.avg[key] = self.sum[key] / self.count[key]
