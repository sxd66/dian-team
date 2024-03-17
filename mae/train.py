from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from dataload import data_load
from models_mae import MaskedAutoencoderViT
from utils import AverageMeter,accuracy,val,save_checkpoint,load_checkpoint



train_loader,val_loader=data_load()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_epoch=20
resume=True
exm=True

model=MaskedAutoencoderViT(img_size=32,patch_size=8,in_chans=1,embed_dim=128,depth=12,
   num_heads=8,decoder_embed_dim=64,decoder_depth=3,decoder_num_heads=4,mlp_ratio=3)
optimer=torch.optim.Adam(model.parameters(),0.01)
#lost_fun=nn.CrossEntropyLoss()
epoch=0
if(resume==True):
    state=load_checkpoint('ckpt2')
    epoch = state["epoch"] + 1

    model.load_state_dict(state["model"])
model.train()
while epoch<max_epoch:
    phar = tqdm(range(len(train_loader)))
    avg = AverageMeter()

    for i,(image,y) in enumerate(train_loader):

        model.to(device)
        image=image.to(device)

        optimer.zero_grad()


        lost,pred,mask=model(image)
        pred=pred.reshape(32,4,4,8,8).permute(0,1,3,2,4).reshape(32,1,32,32)
        if(exm==True):
            obj={
                "pred":pred,
                "Gt_image":image
            }
            with open("img", "wb") as f:
                torch.save(obj, f)
                break
        avg.update(lost)
        lost.backward()
        optimer.step()


        phar.set_description("train:::epoch:{}  lost:{:.3f} ".format(epoch,avg.avg))
        phar.update()
    phar.close()
    if((epoch)%2==0):
        val(model,val_loader)
        state={
            'epoch':epoch,
            'model':model.state_dict()
        }
        save_checkpoint(state,'ckpt2')
    epoch+=1
