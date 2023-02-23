
from torch.nn import Sequential, Conv2d
from torch import rand

from . import residual_bottleneck


def Model(M: int, N: int) -> Sequential:
    return Sequential(
        Conv2d(3, N, kernel_size=5, stride=2, padding=2),
        residual_bottleneck.Model(N),
        Conv2d(N, N, kernel_size=5, stride=2, padding=2),
        residual_bottleneck.Model(N),
        Conv2d(N, N, kernel_size=5, stride=2, padding=2),
        residual_bottleneck.Model(N),
        Conv2d(N, M, kernel_size=5, stride=2, padding=2),
    )


if __name__ == "__main__":
    test_data = rand((1, 3, 1024, 1024))
    model = Model(192, 192)
    y_hat = model(test_data)
    print(test_data.shape)
    print(y_hat.shape)
