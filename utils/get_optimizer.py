from torch.optim import Adam
from opt import opt


def get_optimizer(net):
    if opt.freeze:

        for p in net.parameters():
            p.requires_grad = True
        for q in net.backbone.parameters():
            q.requires_grad = False

        optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=opt.lr, weight_decay=5e-4,
                         amsgrad=True)

    else:


        optimizer = Adam(net.parameters(), lr=opt.lr, weight_decay=5e-4, amsgrad=True)

    return optimizer
