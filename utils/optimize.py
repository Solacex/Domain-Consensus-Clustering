import torch.optim as optim
import torch.nn.functional as F


def inverseDecaySheduler(optimizer, step, initial_lr, gamma=10, power=0.75, num_steps=1000):
    '''
    From EasyDL Library: https://github.com/thuml/easydl/blob/master/easydl/common/scheduler.py

    change as initial_lr * (1 + gamma * min(1.0, iter / max_iter) ) ** (- power)
    as known as inv learning rate sheduler in caffe,
    see https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
    the default gamma and power come from <Domain-Adversarial Training of Neural Networks>
    code to see how it changes(decays to %20 at %10 * max_iter under default arg)::
        from matplotlib import pyplot as plt
        ys = [inverseDecaySheduler(x, 1e-3) for x in range(10000)]
        xs = [x for x in range(10000)]
        plt.plot(xs, ys)
        plt.show()
    '''

    lr = initial_lr * ((1 + gamma * min(1.0, step / float(num_steps))) ** (- power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
#    return initial_lr * ((1 + gamma * min(1.0, step / float(num_steps))) ** (- power))

def adjust_learning_rate_inv(lr, optimizer, iters, alpha=0.001, beta=0.75):
    lr = lr / pow(1.0 + alpha * iters, beta)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
#    for param_group in optimizer.param_groups:
 #       param_group['lr'] = lr * param_group['lr_mult']
