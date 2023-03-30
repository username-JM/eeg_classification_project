import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_data()
        self.torch_form()

    def load_data(self):
        s = self.args.train_subject[0]
        if self.args.phase == 'train':
            self.X = np.load(f"./data/S{s:02}_X_train.npy")
            self.y = np.load(f"./data/S{s:02}_y_train.npy")
        else:
            self.X = np.load(f"./data/S{s:02}_X_test.npy")
            self.y = np.load(f"./answer/S{s:02}_y_test.npy")
        if len(self.X.shape) <= 3:
            self.X = np.expand_dims(self.X, axis=1)

    def torch_form(self):
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = [self.X[idx], self.y[idx]]
        return sample



def data_loader(args):
    print("[Load data]")
    # Load train data
    args.phase = "train"
    trainset = CustomDataset(args)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Load val data
    args.phase = "val"
    valset = CustomDataset(args)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Print
    print(f"train_set size: {train_loader.dataset.X.shape}")
    print(f"val_set size: {val_loader.dataset.X.shape}")
    print("")
    return train_loader, val_loader
