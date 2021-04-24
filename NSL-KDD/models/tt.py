import torch
from cgan import *


X = torch.normal(0,1,(100000,31))
Y = torch.randint(0,2,(100000,1))
print(Y.shape)
args = [32, 10, 20, 0.0001]
model = CGAN(args,X,Y)
model.train()

pred = model.generate_data(torch.randint(0,2,(10,1)))
print(pred)
dataset = TensorDataset(X,Y)
torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False)
