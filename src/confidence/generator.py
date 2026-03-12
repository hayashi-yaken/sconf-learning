import torch
from src.models import mlp_model
from src.losses import logistic
from src.device import device


def confidence_generator(train_loader, v_train_loader):
    """Train an auxiliary MLP with true labels and return per-sample confidence scores p_i."""
    model = mlp_model(input_dim=28 * 28, hidden_dim=500, output_dim=1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(10):
        for i, (images, label) in enumerate(train_loader):
            images, label = images.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = logistic(f=outputs, label=label)
            loss.backward()
            optimizer.step()
    conf = torch.Tensor([]).to(device)
    for i, (images, label) in enumerate(v_train_loader):
        images, label = images.to(device), label.to(device)
        a = model(images).detach().to(device)
        c = torch.sigmoid(a).to(device)
        conf = torch.cat((conf, c), 0).to(device)
    return conf
