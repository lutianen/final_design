from torchvision.datasets import CIFAR10
from torchvision import transforms as transforms
from torch.utils.data import DataLoader

class Data:
    def __init__(self, args) -> None:
        pin_memory = True

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]),
        ])

        trainset = CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
        self.train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)

        testset = CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
        self.test_loader = DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
