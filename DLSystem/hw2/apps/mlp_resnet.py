import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    module = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )

    model = nn.Sequential(
        nn.Residual(module),
        nn.ReLU()
    )

    return model


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    blocks = []
    blocks.append(nn.Linear(dim, hidden_dim))
    blocks.append(nn.ReLU())
    resi_dim = hidden_dim // 2
    for _ in range(num_blocks):
        blocks.append(ResidualBlock(hidden_dim, resi_dim, norm, drop_prob))
    blocks.append(nn.Linear(hidden_dim, num_classes))
    
    return nn.Sequential(*blocks)




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    if opt is None:
        model.eval()
    else:
        model.train()
    
    total_error = 0
    total_loss = 0
    total_num = 0
    for i, (x, y) in enumerate(dataloader):
        x = x.reshape((x.shape[0], -1))
        # y = y.reshape((y.shape[0], -1))

        y_pred = model(x)
        # print("y_pred: ", y_pred.shape)
        # print("y: ", y.shape)
        loss = nn.SoftmaxLoss()(y_pred, y)

        total_error += (y_pred.numpy().argmax(axis=1) != y.numpy()).sum()
        total_loss += loss.numpy()

        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()

        total_num += x.shape[0]

    return total_error / total_num, total_loss / (i + 1)

def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)

    train_dataset = ndl.data.MNISTDataset(
        data_dir + "/train-images-idx3-ubyte.gz",
        data_dir + "/train-labels-idx1-ubyte.gz",
    )
    train_dataloader = ndl.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = ndl.data.MNISTDataset(
        data_dir + "/t10k-images-idx3-ubyte.gz", data_dir + "/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = ndl.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    model = MLPResNet(784, hidden_dim=hidden_dim, num_blocks=3, num_classes=10)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)
    
    test_error, test_loss = epoch(test_dataloader, model, opt)

    return train_error, train_loss, test_error, test_loss




if __name__ == "__main__":
    train_mnist(data_dir="../data")
