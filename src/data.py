from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from pathlib import Path
import kagglehub
import shutil

def download_data(dataset_path) :
    DATASET_DIR = Path(dataset_path)
    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):

        downloaded_path = Path(
            kagglehub.dataset_download("jessicali9530/celeba-dataset")
        )

        src = downloaded_path / "img_align_celeba"

        if not src.exists():
            raise FileNotFoundError("img_align_celeba introuvable dans le dataset téléchargé")

        shutil.copytree(src, DATASET_DIR, dirs_exist_ok=True)


def load_data(dataroot, dataset_size, image_size, batch_size, workers) :
    full_dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])) 


    indices = range(dataset_size)
    dataset = Subset(full_dataset, indices)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    
    return dataloader