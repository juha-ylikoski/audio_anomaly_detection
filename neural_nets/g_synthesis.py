

from torch.nn import Sequential, ConvTranspose2d

from . import attention_module, residual_bottleneck

# TODO figure out conv2d channels


def Model():
    return Sequential(
        attention_module.Model(),
        ConvTranspose2d(1, 1, kernel_size=5, stride=2),
        residual_bottleneck.Model(),
        ConvTranspose2d(1, 1, kernel_size=5, stride=2),
        attention_module.Model(),
        residual_bottleneck.Model(),
        ConvTranspose2d(1, 1, kernel_size=5, stride=2),
        residual_bottleneck.Model(),
        ConvTranspose2d(1, 1, kernel_size=5, stride=2),
    )
