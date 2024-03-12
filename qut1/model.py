import torch.nn as nn
import torch
class Simple_lin(nn.Module):
    def __init__(self):
        super(Simple_lin, self).__init__()
        self.linears=nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self,x):
        x=x.flatten(1,-1)
        pred=self.linears(x)
        return pred

if __name__=="__main__":
    x=torch.randn(32,1,28,28)
    model=Simple_lin()
    y=model(x)
    print(y.shape[0:])
