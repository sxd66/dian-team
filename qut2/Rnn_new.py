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
        self.linears=nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(num_layer-1)])
        self.tanh=nn.Tanh()

    def forward(self,input,hid_layer):
        input=input.transpose(0,1)
        hid_layer=hid_layer.transpose(0,1)
        Seq,Batch,C=input.shape
        hid_layer=hid_layer.unsqueeze(0).repeat(Seq+1,1,1,1)
        output=[]
        for i in range(Seq):
            x=input[i].squeeze(0)
            x=torch.cat((x,hid_layer[i][0]),dim=1)
            temp=self.tanh(self.fc1(x))
            for j,l in enumerate(self.linears):
                hid_layer[i+1,j+1] = l(hid_layer[ i,j+1])
                temp=self.tanh( hid_layer[i+1, j+1] +temp )

            out=self.fc2(temp)
            out=out.relu()
            out=out.softmax(dim=-1)
            output.append(out)
        output=torch.stack(output,dim=0)
        return output

if __name__=="__main__":
    x1=torch.randn(64,28,28)
    hid=torch.randn(64,3,40)
    model=Rnn(28,40,10,3)
    y=model(x1,hid)
    print(y.shape[0:])