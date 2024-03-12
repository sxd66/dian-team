import torch
from tqdm import tqdm
import torch.nn as  nn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output,target):
    _, pred = output.topk(1, 1)
    pred = pred.reshape(1, -1)
    correct = pred.eq(target.reshape(1, -1))
    return correct.sum() / correct.shape[1]
def val(model,dataload):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss=nn.CrossEntropyLoss()
    with torch.no_grad():
        phar = tqdm(range(len(dataload)))
        avg = AverageMeter()
        acc=AverageMeter()
        for i, (image, y) in enumerate(dataload):
            image = image.to(device)
            y = y.to(device)


            pred = model(image)

            lost=loss(pred,y)
            avg.update(lost)
            acc.update(accuracy(pred,y))


            phar.set_description(" lost:{:.3f} acc:{:.3f}".format( avg.avg, acc.avg))
            phar.update()
        phar.close()
def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
if __name__=="__main__":
    x=torch.randn(32,10)
    target=torch.randn(32,1)
    _,pred=x.topk(1,1)
    pred=pred.reshape(1,-1)
    correct = pred.eq(target.reshape(1, -1))
    print(correct.sum()/correct.shape[0])