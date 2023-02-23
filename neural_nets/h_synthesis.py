

from torch import rand
from torch.nn import Sequential, ReLU, ConvTranspose2d


def Model(N):
    """
    Refer to compressAI/models/google/ minnen2018 h_s
    """
    return Sequential(
        ConvTranspose2d(N, N, kernel_size=5, stride=2,
                        padding=2, output_padding=1),
        ReLU(),
        ConvTranspose2d(N, N, kernel_size=5, stride=2,
                        padding=2, output_padding=1),
        ReLU(),
        ConvTranspose2d(N, 2*N, kernel_size=3, stride=1, padding=1)
    )


if __name__ == "__main__":
    test_data = rand((1, 192, 16, 16))
    model = Model(192)
    y_hat = model(test_data)
    print(test_data.shape)
    print(y_hat.shape)
