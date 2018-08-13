from torch.utils.data import DataLoader
from torchvision import transforms

from data.datasets import CarsDataset, CustomDataset


def get_data_loader(opt):
    if opt.is_train:
        transform = transforms.Compose([
            transforms.Resize([opt.width_size + 32, opt.width_size + 32]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(opt.width_size, scale=(0.4, 0.95)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize([opt.width_size + 32, opt.width_size + 32]),
            transforms.CenterCrop(opt.width_size),
            transforms.ToTensor(),
        ])

    if opt.dataset == 'CARS':
        if opt.is_train:
            train_set = CarsDataset(
                root_dir=opt.data_dir,
                train=True,
                transform=transform,
            )
            return DataLoader(
                train_set,
                batch_size=opt.small_batch_size,
                shuffle=True,
                num_workers=opt.num_preprocess_workers,
                drop_last=True
            )
        else:
            test_set = CarsDataset(
                root_dir=opt.data_dir,
                train=False,
                transform=transform,
            )
            return DataLoader(
                test_set,
                batch_size=opt.small_batch_size,
                shuffle=False,
                num_workers=opt.num_preprocess_workers,
            )
    elif opt.dataset == 'Custom':
        train_set = CustomDataset(
            transform=transform,
        )
        return DataLoader(
            train_set,
            batch_size=opt.small_batch_size,
            shuffle=True,
            num_workers=opt.num_preprocess_workers,
            drop_last=True
        )
