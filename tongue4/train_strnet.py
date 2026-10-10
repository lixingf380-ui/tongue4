import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.v7_dataset import V7TongueDataset
from models.strnet import STRNet

device="cuda:1" if torch.cuda.is_available() else "cpu"
train=V7TongueDataset("dataset_v7_continuous_synthetic/windows/window128_train_generatorA_mixed.npz")
loader=DataLoader(train,128,shuffle=True)

model=STRNet().to(device)
opt=torch.optim.AdamW(model.parameters(),1e-4)
ce=nn.CrossEntropyLoss()
mse=nn.MSELoss()

for e in range(50):
    model.train()
    for x,y,q in loader:
        x,y,q=x.to(device),y.to(device),q.to(device)
        opt.zero_grad()
        out,r=model(x)
        loss=ce(out,y)+0.3*mse(r,q)
        loss.backward()
        opt.step()
    torch.save(model.state_dict(),"best_strnet.pth")
    print("epoch",e)
