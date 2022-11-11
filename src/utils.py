
from torchvision.utils import make_grid
import numpy as np
import math
import torchvision.transforms as transforms
import config
import torch
import matplotlib.pyplot as plt

class Data_load:
    def __init__(self,url,mode='train',trans=None):
        data=np.loadtxt(url,delimiter=',',skiprows=1,dtype=np.float32)
        A=data[:,1:][data[:,0]==0]
        B=data[:,1:][data[:,0]==1]
        A=A.reshape(A.shape[0],28,28)/255
        B=B.reshape(B.shape[0],28,28)/255
        
        if A.shape > B.shape:
            limit= B.shape[0]
        else:
            limit=A.shape[0]
        index=int(math.floor(limit*(4/5)))
        if mode=='train': 
            self.A=A[:index,:]
            self.B=B[:index,:]
        elif mode=='test':
            self.A=A[index:limit,:]
            self.B=B[index:limit,:]
        
        
        self.len=self.A.shape[0]    
        self.trans=trans
        
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        A=self.A[index]
        B=self.B[index]
        if self.trans is not None:
            A=self.trans(A)
            B=self.trans(B)
        return A,B
    

transformations = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(0.5, 0.5)
                                      ])


def plot(real_a, real_b):
    """plots the generated images

    Args:
        real_a (tensor): real image a
        real_b (tensor): real image b
    """
    real_a = real_a.to(config.device)
    real_b = real_b.to(config.device)
    G_ab.eval()
    G_ba.eval()
    with torch.no_grad():
        fake_a = G_ba(real_b[:5]).detach()
        fake_b = G_ab(real_a[:5]).detach()

        f_a = make_grid(fake_a)
        f_b = make_grid(fake_b)
        grid = torch.cat((f_a, f_b), 1).to('cpu')
        grid = transforms.functional.to_pil_image(grid)
#         grid=grid
        plt.figure(figsize=(8, 8))
        plt.matshow(grid)
        plt.axis('off')
        plt.show()
