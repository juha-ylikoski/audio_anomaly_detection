
from torch.nn import Sequential, Conv2d, ReLU

# TODO figure out conv2d channels


def Model():
    return Sequential(
        Conv2d(1, 1, kernel_size=3, stride=1),
        ReLU(),
        Conv2d(1, 1, kernel_size=5, stride=2),
        ReLU(),
        Conv2d(1, 1, kernel_size=5, stride=2)
    )
