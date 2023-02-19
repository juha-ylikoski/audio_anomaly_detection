

from torch.nn import Sequential, ReLU, ConvTranspose2d

# TODO figure out conv2d channels


def Model():
    return Sequential(
        ConvTranspose2d(1, 1, kernel_size=5, stride=2),
        ReLU(),
        ConvTranspose2d(1, 1, kernel_size=5, stride=2),
        ReLU(),
        ConvTranspose2d(1, 1, kernel_size=3, stride=1)
    )
