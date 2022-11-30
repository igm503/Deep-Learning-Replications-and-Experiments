import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt


def elastic_transform(image, alpha_range=48, sigma=5, p=.5, random_state=None):
    if p > random.random():
        if random_state is None:
            random_state = np.random.RandomState(None)

        if np.isscalar(alpha_range):
            alpha = alpha_range
        else:
            alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

        shape = image.shape
        dx = gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(
            shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy,
                                                        (-1, 1)), np.reshape(z, (-1, 1))
        return torch.Tensor(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape))
    return image


def MNIST_dataloaders(data_augment, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0, 1)])
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0, 1), torchvision.transforms.Lambda(elastic_transform)])
    # Training Data
    trainset = [torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                           transform=train_transform) for i in range(data_augment)]
    trainset = torch.utils.data.ConcatDataset(trainset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2, persistent_workers=True)
    # Eval Data
    test_data = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=128, shuffle=False, num_workers=2, persistent_workers=True)

    return train_loader, test_loader


def EMNIST_dataloaders(data_augment, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0, 1), ])
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0, 1), torchvision.transforms.Lambda(elastic_transform)])
    # Training Data
    trainset = [torchvision.datasets.EMNIST(root='./data', split='digits', train=True, download=True,
                                            transform=train_transform) for i in range(data_augment)]
    trainset = torch.utils.data.ConcatDataset(trainset)
    print(len(trainset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2, persistent_workers=True)
    # Eval Data
    test_data = torchvision.datasets.EMNIST(
        root='./data', split='digits', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=128, shuffle=False, num_workers=2, persistent_workers=True)

    return train_loader, test_loader


if __name__ == '__main__':
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0, 1)])  # , torchvision.transforms.Lambda(elastic_transform)])
    trainset = [torchvision.datasets.EMNIST(root='./data', split='digits', train=True, download=True,
                                            transform=train_transform) for i in range(1)]

    num = 899
    img = np.array(trainset[0][num][0].squeeze())
    img_t = np.array(elastic_transform(trainset[0][num][0], p=1).squeeze())
    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax = ax.flatten()
    ax[0].imshow(img, interpolation='nearest')
    ax[1].imshow(img_t, interpolation='nearest')
    ax[2].imshow(np.array(elastic_transform(trainset[0][num]
                 [0], p=1).squeeze()), interpolation='nearest')
    ax[3].imshow(np.array(elastic_transform(trainset[0][num]
                 [0], p=1).squeeze()), interpolation='nearest')
    plt.show()
