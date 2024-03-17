import torch
import torch.nn as nn
from qut3.attention import Muilt_atten

class Mlp(nn.Module):
    def __init__(self,emb_dim,mlp_ratio):
        super(Mlp, self).__init__()
        self.linear1=nn.Linear(emb_dim,int(emb_dim*mlp_ratio))
        self.linear2=nn.Linear(int(emb_dim*mlp_ratio),emb_dim)
        self.norm=nn.LayerNorm(emb_dim)
    def forward(self,x):
        temp=self.linear1(x).relu()
        x= self.norm( x+self.linear2(temp).relu())
        return x
class Block(nn.Module):
    def __init__(self,dim, heads, mlp_ratio, qkv_bias, norm_layer):
        super(Block, self).__init__()
        self.mlp=Mlp(dim,mlp_ratio)
        self.atten=Muilt_atten(heads,dim)
        self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        temp=self.atten(x)
        x=x+self.norm(temp[0])
        x=x+self.norm(self.mlp(x))
        return x

if __name__=='__main__':
    x=torch.randn(32,1,16)
    model=Block(16,4,3,1,1)
    output=model(x)

    print(output)
