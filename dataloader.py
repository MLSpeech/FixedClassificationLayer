import torch
import torchvision
import torchvision.transforms as transforms

def stl_loader(path, batch_size, num_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.STL10(root=path, split='train', transform=transform_train, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.STL10(root=path, split='test', transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def load_stl(data_path,batch_size, num_workers):
    return stl_loader(data_path,batch_size, num_workers)
