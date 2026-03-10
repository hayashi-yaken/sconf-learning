import torch.nn as nn


class Siamese_net(nn.Module):
    def __init__(self, model):
        super(Siamese_net, self).__init__()
        self.fc = model

    def sub_forward(self, x):
        out = self.fc(x)
        return out

    def forward(self, x1, x2):
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)
        return h1, h2
