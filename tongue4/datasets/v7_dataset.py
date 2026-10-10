import numpy as np
import torch
from torch.utils.data import Dataset

class V7TongueDataset(Dataset):
    def __init__(self, npz_path, use_event_only=True):
        data=np.load(npz_path)
        self.x=data["x"].astype("float32")
        self.y=data["y_command10"].astype("int64")
        self.q=data["target_quality"].astype("float32")
        if use_event_only:
            m=data["is_event"]==1
            self.x,self.y,self.q=self.x[m],self.y[m],self.q[m]

    def __len__(self): return len(self.y)

    def __getitem__(self,i):
        x=np.transpose(self.x[i],(1,2,0)).reshape(18,128)
        return torch.tensor(x), torch.tensor(self.y[i]), torch.tensor(self.q[i])
