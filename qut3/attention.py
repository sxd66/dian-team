import torch
import torch.nn as nn

def attention(q,k,v,mask=None):
    B,N,C=q.shape
    attn=(q@k.transpose(1,2))*(C**-0.5)
    if(mask is not None):
        attn=attn*mask
    attn=attn.softmax(dim=-1)
    out=attn@v
    return out,attn

class Muilt_atten(nn.Module):
    def __init__(self,head,embed):
        super(Muilt_atten, self).__init__()
        self.head=head
        self.embed=embed
        self.qkv=nn.Linear(embed,3*embed)
        self.norm=nn.LayerNorm(self.embed//self.head)
    def forward(self,x):
        B,N,C=x.shape
        x=self.qkv(x).reshape(B,N,3,self.head,self.embed//self.head).permute(2,0,3,1,4)
        q,k,v=x[0],x[1],x[2]
        attn=q@k.transpose(-1,-2)*((self.embed//self.head)**-0.5)
        attn=attn.softmax(dim=-1)
        output=self.norm(attn@v)
        output=output.permute(0,2,1,3).reshape(B,N,C)
        return output,attn




class Group_atten(nn.Module):
    def __init__(self,head,embed,group=1):
        super(Group_atten, self).__init__()
        self.head=head
        self.embed=embed
        self.new_dim=self.embed//self.head
        self.qkv=nn.Linear(embed,embed+group*self.new_dim*2)
        self.group=group
        self.gp_head=self.head//self.group
    def forward(self,x):
        B,N,C=x.shape
        x=self.qkv(x).split([self.embed,self.group*self.new_dim,self.group*self.new_dim],dim=2)
        q,k,v=x[0],x[1],x[2]
        q=q.reshape(B,N,self.head,self.new_dim).transpose(1,2)
        k=k.reshape(B,N,self.group,self.new_dim).transpose(1,2)
        v=v.reshape(B,N,self.group,self.new_dim).transpose(1,2)
        k_list=[k[:,i].unsqueeze(1).expand(B,self.gp_head,N,self.new_dim) for i in range(self.group)]
        k=torch.concatenate(k_list,dim=1)
        v_list=[v[:,i].unsqueeze(1).expand(B,self.gp_head,N,self.new_dim) for i in range(self.group)]
        v=torch.concatenate(v_list,dim=1)

        attn = q @ k.transpose(-1, -2) * ((self.embed // self.head) ** -0.5)
        attn = attn.softmax(dim=-1)
        output = attn @ v
        output = output.permute(0, 2, 1, 3).reshape(B, N, C)

        return output,attn

if __name__=='__main__':
    x=torch.randn(32,20,16)
    model=Muilt_atten(4,16)
    output, attn=model(x)

    print(attn[0][0])
























