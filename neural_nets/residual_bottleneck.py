

from torch.nn import Module, Sequential, Conv2d, ReLU


class Model(Module):
    def __init__(self, N) -> None:
        super().__init__()
        self.seq = Sequential(
            Conv2d(N, N, kernel_size=1, stride=1),
            ReLU(),
            Conv2d(N, N, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(N, N, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return self.seq(x) + x
