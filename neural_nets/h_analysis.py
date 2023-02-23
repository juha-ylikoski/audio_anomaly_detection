
from torch import rand
from torch.nn import Sequential, Conv2d, ReLU


def Model(N):
    """
    Refer to compressAI/models/google/ minnen2018 h_a
    """
    return Sequential(
        Conv2d(N, N, kernel_size=3, stride=1, padding=1),
        ReLU(),
        Conv2d(N, N, kernel_size=5, stride=2, padding=2),
        ReLU(),
        Conv2d(N, N, kernel_size=5, stride=2, padding=2)
    )


if __name__ == "__main__":
    test_data = rand((1, 192, 64, 64))
    model = Model(192)
    y_hat = model(test_data)
    print(test_data.shape)
    print(y_hat.shape)
