import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
with open("img","rb") as f:
    temp=torch.load(f,map_location="cpu")
    pred,Gt=temp['pred'],temp['Gt_image']
    pred=pred.permute(0,2,3,1).detach().numpy()
    Gt=Gt.permute(0,2,3,1).detach().numpy()
    list2=[]
    list3=[]
    for i in range(pred.shape[0]):
        list2.append(pred[i])
        list3.append(Gt[i])
    pred=np.concatenate(list2,axis=1)
    Gt=np.concatenate(list3,axis=1)
    cv2.imshow("pred",pred)

    cv2.imshow("Gt_image",Gt)
    cv2.waitKey(0)















