import torch.nn as nn
from torch.utils.data import Dataset
import pickle

def open_pickle(path):
    try:
        with open(path,'rb') as f:
            data = pickle.load(f)
    except:
        print(path)
    return data

def save_pickle(path, data):
  with open(path,'wb') as f:
    pickle.dump(data,f)

def weights_init(m, what_init='xavier'):
    if what_init == 'xavier':
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
    if what_init == 'kaiming':
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight)        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)       
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        
class AKIDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y, idx    