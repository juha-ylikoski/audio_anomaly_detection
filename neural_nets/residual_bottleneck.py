

from torch.nn import Module, Sequential, Conv2d, ReLU


# TODO figure out conv2d channels


class PartialModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = Sequential(
            Conv2d(1, 1, kernel_size=1, stride=1),
            ReLU(),
            Conv2d(1, 1, kernel_size=3, stride=1),
            ReLU(),
            Conv2d(1, 1, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return self.seq(x) + x


def Model():
    return Sequential(
        PartialModel(),
        PartialModel(),
        PartialModel()
    )
