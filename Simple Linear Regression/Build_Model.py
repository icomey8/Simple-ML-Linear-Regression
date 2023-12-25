import torch
from torch import nn
import matplotlib.pyplot as plt



### BUILDING OUR MODEL
weight = 0.3
bias = 0.9

X = torch.arange(0,1,0.01).unsqueeze(dim=1)
y = weight * X + bias

data_split = int(0.8 * len(X))
X_train, y_train = X[:data_split], y[:data_split]
X_test, y_test =  X[data_split:], y[data_split:]

def plot_input(training_data=X_train, training_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    plt.figure(figsize=(10,5))
    plt.scatter(training_data, training_labels, c="r", s=4, label="training data")
    plt.scatter(test_data, test_labels, c="b", s=4, label="test data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c='purple', s=4, label="predictions")
    plt.legend(prop={"size":10})
    plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                dtype=torch.float,
                                                requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1,
                                             dtype=torch.float,
                                             requires_grad=True))
    def forward(self, X):
        return self.weights * X + self.bias


