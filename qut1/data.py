from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
from model import Simple_lin
from utils import AverageMeter,accuracy,val,save_checkpoint,load_checkpoint
# 定义数据转换
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.2),

    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform_val = transforms.Compose([


    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# 加载训练集和测试集
train_dataset = datasets.MNIST(root="../data/mnist", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="../data/mnist", train=False, transform=transform_val, download=True)

train_loader=DataLoader(train_dataset,32,True)
val_loader=DataLoader(test_dataset,32,True)
transform2=transforms.Compose([
    transforms.ToPILImage()
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_epoch=20
resume=True

model=Simple_lin()
optimer=torch.optim.Adam(model.parameters(),0.01)
lost_fun=nn.CrossEntropyLoss()
epoch=0
if(resume==True):
    state=load_checkpoint('ckpt')
    epoch = state["epoch"] + 1

    model.load_state_dict(state["model"])
model.train()
while epoch<max_epoch:
    phar = tqdm(range(len(train_loader)))
    avg = AverageMeter()
    acc=AverageMeter()
    for i,(image,y) in enumerate(train_loader):

        model.to(device)
        image=image.to(device)
        y=y.to(device)
        optimer.zero_grad()

        pred=model(image)

        lost=lost_fun(pred,y)
        acc.update(accuracy(pred,y))
        avg.update(lost)
        lost.backward()
        optimer.step()


        phar.set_description("epoch:{}  lost:{:.3f} acc:{:.3f}".format(epoch,avg.avg,acc.avg))
        phar.update()
    phar.close()
    if((epoch)%2==0):
        val(model,val_loader)
        state={
            'epoch':epoch,
            'model':model.state_dict()
        }
        save_checkpoint(state,'ckpt')
    epoch+=1



