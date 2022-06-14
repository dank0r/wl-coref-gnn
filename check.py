import torch
import numpy as np

print(torch.cuda.is_available())
device = torch.device('cuda') 
a = torch.Tensor(np.eye(100)).to(device)
b = torch.Tensor(np.eye(100)+1).to(device)
c = torch.matmul(a, b)
print(c)
