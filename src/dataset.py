from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    train_samples: int | None = None,
    val_samples: int | None = None,
):
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )

    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    if train_samples is not None:
        train_samples = min(train_samples, len(train_dataset))
        train_dataset = Subset(train_dataset, range(train_samples))

    if val_samples is not None:
        val_samples = min(val_samples, len(val_dataset))
        val_dataset = Subset(val_dataset, range(val_samples))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader
