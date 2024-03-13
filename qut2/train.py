from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from dataload import data_load
from Rnn import Rnn
from utils import AverageMeter,accuracy,val,save_checkpoint,load_checkpoint



train_loader,val_loader=data_load()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_epoch=20
resume=False

model=Rnn(28,20,10,0)
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
        image=image.to(device).squeeze(1)
        y=y.to(device)
        optimer.zero_grad()

        hid = torch.zeros(32, 1, 20)
        hid=hid.to(device)
        pred=model(image,hid)

        pred=pred[-1]
        lost=lost_fun(pred,y)
        acc.update(accuracy(pred,y))
        avg.update(lost)
        lost.backward()
        optimer.step()


        phar.set_description("train:::epoch:{}  lost:{:.3f} acc:{:.3f}".format(epoch,avg.avg,acc.avg))
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