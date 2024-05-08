from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

def get_dataset(dataset_path, tfs=None):
    """
    make dataset for training and validation
    """
    print("[INFO] loading datasets...")
    if not tfs:
        tfs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    dataset = ImageFolder(root=f"{dataset_path}",
            transform=tfs)
    print("[INFO] loaded dataset contains {} samples...".format(
            len(dataset)))
    return dataset

def get_dataloader(dataset, batch_size):
    """
    make dataloaders
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
