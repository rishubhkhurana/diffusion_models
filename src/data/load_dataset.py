from datasets import load_dataset
from torch.utils.data import Dataset


class HFDataset(Dataset):
    def __init__(self, name: str = 'fashion_mnist', mode: str = 'train'):
        super().__init__()
        self.ds = load_dataset(name)[mode]
    
    def __getitem__(self, idx):
        return self.ds[idx]

    def __len__(self):
        return len(self.ds)


