import torch
import torch.nn.functional as F

from src.device import device


def accuracy_check(loader, model):
    sig = F.sigmoid
    num_samples = 0
    total = torch.zeros(1).to(device)
    for images, labels in loader:
        labels, images = labels.to(device), images.to(device)
        outputs = model(images).detach()
        sig_outputs = sig(outputs)
        t1 = (
            ((labels == 1).view(len(labels), -1) & (sig_outputs > 0.5).view(len(labels), -1)) |
            ((labels == -1).view(len(labels), -1) & (sig_outputs < 0.5).view(len(labels), -1))
        )
        total += torch.sum(t1)
        num_samples += labels.size(0)
    return 100 * total / num_samples


def check_sia(test_loader, fp, fn, model):
    num = 0.00
    tot = 0.00
    for i, (image, label) in enumerate(test_loader):
        image = image.to(device)
        f = model.sub_forward(image).to(device)
        tot = tot + len(f)
        fp = fp.to(device)
        fn = fn.to(device)
        fp = fp.view(len(fp), -1)
        fn = fn.view(len(fn), -1)
        for j in range(len(f)):
            ff = f[j]
            ff = ff.view(len(ff), -1)
            d1 = F.pairwise_distance(ff, fp).to(device)
            d2 = F.pairwise_distance(ff, fn).to(device)
            d11 = torch.sum(torch.mul(d1, d1))
            d22 = torch.sum(torch.mul(d2, d2))
            if d11 > d22:
                l = -1
            else:
                l = 1
            if l == label[j]:
                num = num + 1
    if num > tot - num:
        ans = num
    else:
        ans = tot - num
    return ans / tot
