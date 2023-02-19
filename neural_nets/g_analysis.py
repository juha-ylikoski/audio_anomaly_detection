
from torch.nn import Sequential, Conv2d, ReLU

from . import attention_module, residual_bottleneck

# TODO figure out conv2d channels


def Model(out_channels, N):
    return Sequential(
        Conv2d(1, 1, kernel_size=5, stride=2),
        residual_bottleneck.Model(),
        Conv2d(1, 1, kernel_size=5, stride=2),
        residual_bottleneck.Model(),
        attention_module.Model(),
        Conv2d(1, 1, kernel_size=5, stride=2),
        residual_bottleneck.Model(),
        Conv2d(1, out_channels, kernel_size=5, stride=2),
        attention_module.Model(),
    )
