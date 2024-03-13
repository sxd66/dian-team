from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import numpy as np

class Rnn(nn.Module):
    def __init__(self,input,hidden,output,num_layer):
        super(Rnn, self).__init__()
        self.hidden=hidden
        self.input=input
        self.output=output
        self.num_layer=num_layer
        self.fc1=nn.Linear(hidden+input,hidden)
        self.fc2=nn.Linear(hidden,output)
        self.linears=nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(num_layer)])
        self.tanh=nn.Tanh()

    def forward(self,input,hid_layer):
        input=input.transpose(0,1)
        hid_layer=hid_layer.transpose(0,1).squeeze(0)
        Seq,Batch,C=input.shape
        output=[]
        for i in range(Seq):
            x=input[i].squeeze(0)
            x=torch.cat((x,hid_layer),dim=1)
            hid_layer=self.tanh(self.fc1(x))
            for l in self.linears:
                hid_layer=(l(hid_layer)+hid_layer).relu()
            out=self.fc2(hid_layer)
            out=out.relu()
            out=out.softmax(dim=-1)
            output.append(out)
        output=torch.stack(output,dim=0)
        return output

if __name__=="__main__":
    x1=torch.randn(64,28,28)
    hid=torch.randn(64,1,20)
    model=Rnn(28,20,10,2)
    y=model(x1,hid)
    print(y.shape[0:])










